"""Top-level packer. Converts a Python callable into a provider-shaped dict."""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Callable, Iterable
from typing import Any, get_type_hints

from agent_tool_spec_pack.docstring_parsers import parse_docstring
from agent_tool_spec_pack.type_mapping import to_schema
from agent_tool_spec_pack.types import Format, InspectionResult, ParamInfo

# Parameters we never want to expose to an LLM (implicit instance / collector args).
_SKIP_PARAMS = {"self", "cls"}
_SKIP_KINDS = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}


def pack(
    fn: Callable[..., Any],
    format: Format | str = Format.ANTHROPIC,  # noqa: A002 - "format" reads naturally for callers
) -> dict[str, Any]:
    """Convert `fn` into the provider tool-schema shape requested by `format`."""
    fmt = Format(format) if not isinstance(format, Format) else format
    result = pack_inspect(fn)
    return _shape(result, fmt)


def pack_many(
    fns: Iterable[Callable[..., Any]],
    format: Format | str = Format.ANTHROPIC,  # noqa: A002
) -> list[dict[str, Any]]:
    """Pack many callables in one go, preserving input order."""
    return [pack(f, format=format) for f in fns]


def pack_inspect(fn: Callable[..., Any]) -> InspectionResult:
    """Run the inspection step only. Useful for debugging or building custom shapes."""
    sig = inspect.signature(fn)
    hints = _resolve_hints(fn)

    doc = inspect.getdoc(fn) or ""
    description, param_docs = parse_docstring(doc)

    params: list[ParamInfo] = []
    required: list[str] = []
    skipped: list[str] = []

    for name, param in sig.parameters.items():
        if name in _SKIP_PARAMS:
            skipped.append(name)
            continue
        if param.kind in _SKIP_KINDS:
            warnings.warn(
                f"agent_tool_spec_pack: skipping {param.kind.description} parameter '{name}' on"
                f" {getattr(fn, '__qualname__', getattr(fn, '__name__', repr(fn)))};"
                " LLM tool schemas can't represent *args/**kwargs",
                stacklevel=3,
            )
            skipped.append(name)
            continue

        annotation = hints.get(name, param.annotation)
        schema, is_required_from_type = to_schema(annotation)

        # A parameter description from the docstring is optional but very nice to have.
        desc = (param_docs.get(name) or "").strip()
        if desc:
            schema["description"] = desc

        has_default = param.default is not inspect.Parameter.empty
        if has_default:
            schema["default"] = _json_safe_default(param.default)
        else:
            # Only required when the annotation didn't already say "this can be None".
            if is_required_from_type:
                required.append(name)

        params.append(
            ParamInfo(
                name=name,
                annotation=annotation,
                has_default=has_default,
                default=param.default if has_default else None,
                description=desc,
                schema=schema,
            )
        )

    return InspectionResult(
        name=getattr(fn, "__name__", "anonymous"),
        description=description,
        params=params,
        required=required,
        skipped=skipped,
        raw_doc=doc,
    )


def _resolve_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """Resolve type hints, including PEP 563 forward refs that reference closure cells.

    With `from __future__ import annotations` in effect, annotations are stored
    as strings. `typing.get_type_hints` resolves them against the function's
    module globals, but it does not see names captured from an enclosing
    function. We fall back to building a `localns` dict from the function's
    closure so types like dataclasses or Enums defined inside a test function
    still resolve.
    """
    try:
        return get_type_hints(fn, include_extras=False)
    except Exception:
        pass

    # Try again with closure cells exposed as locals.
    localns: dict[str, Any] = {}
    closure = getattr(fn, "__closure__", None) or ()
    code = getattr(fn, "__code__", None)
    freevars = getattr(code, "co_freevars", ()) if code is not None else ()
    for name, cell in zip(freevars, closure, strict=False):
        try:
            localns[name] = cell.cell_contents
        except ValueError:
            continue
    try:
        return get_type_hints(fn, localns=localns, include_extras=False)
    except Exception:
        return {}


def _json_safe_default(value: Any) -> Any:
    """Best-effort coerce a Python default into a JSON-safe primitive."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_json_safe_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_default(v) for k, v in value.items()}
    # Enum, dataclass, custom object: stringify so the schema stays JSON-encodable.
    return repr(value)


def _build_input_schema(result: InspectionResult) -> dict[str, Any]:
    """Build the inner JSON Schema object that providers nest under different keys."""
    properties = {p.name: p.schema for p in result.params}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(result.required),
    }
    return schema


def _shape(result: InspectionResult, fmt: Format) -> dict[str, Any]:
    description = result.description
    schema = _build_input_schema(result)
    name = result.name

    if fmt in (Format.ANTHROPIC, Format.BEDROCK_ANTHROPIC):
        return {
            "name": name,
            "description": description,
            "input_schema": schema,
        }
    if fmt is Format.OPENAI:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        }
    if fmt is Format.GEMINI:
        return {
            "name": name,
            "description": description,
            "parameters": _to_camel_case(schema),
        }
    if fmt is Format.BEDROCK_LLAMA:
        return {
            "name": name,
            "description": description,
            "parameters": schema,
        }
    raise ValueError(f"Unsupported format: {fmt!r}")  # pragma: no cover - StrEnum guards this


# Keys that Gemini's Schema proto expects in camelCase (everything else is the
# JSON Schema lowercase name). We walk the dict and rename only those keys.
_GEMINI_CAMEL_KEYS = {
    "additional_properties": "additionalProperties",
    "additionalproperties": "additionalProperties",
    "unique_items": "uniqueItems",
    "uniqueitems": "uniqueItems",
    "min_items": "minItems",
    "minitems": "minItems",
    "max_items": "maxItems",
    "maxitems": "maxItems",
    "min_length": "minLength",
    "minlength": "minLength",
    "max_length": "maxLength",
    "maxlength": "maxLength",
    "any_of": "anyOf",
    "anyof": "anyOf",
    "one_of": "oneOf",
    "oneof": "oneOf",
    "all_of": "allOf",
    "allof": "allOf",
}

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _to_camel_case(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            new_key = _gemini_key(k)
            out[new_key] = _to_camel_case(v)
        return out
    if isinstance(value, list):
        return [_to_camel_case(v) for v in value]
    return value


def _gemini_key(key: str) -> str:
    lower = key.lower().replace("-", "_")
    if lower in _GEMINI_CAMEL_KEYS:
        return _GEMINI_CAMEL_KEYS[lower]
    if "_" in key:
        first, *rest = key.split("_")
        return first + "".join(part.title() for part in rest)
    return key
