"""Map Python type hints to JSON Schema fragments.

The packer uses `to_schema(annotation)` to convert a single annotation into a
JSON Schema dict and a "required" flag (Optional types lower the flag).

Supported:
- Primitives: int, float, str, bool, bytes
- Containers: list, tuple, dict, set, frozenset (typed and untyped)
- Optional[T], T | None
- Literal[...]
- Enum subclasses
- Dataclasses (recursive: fields become properties)
- Any / unknown -> {} (no constraints)
"""

from __future__ import annotations

import dataclasses
import inspect
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

# Python 3.10+ types.UnionType lets us spot `X | Y` syntax.
try:  # pragma: no cover - presence depends on Python version
    from types import UnionType  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - very old Pythons; we require 3.10+
    UnionType = None  # type: ignore[assignment]


_PRIMITIVES: dict[type, dict[str, Any]] = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
    bytes: {"type": "string", "format": "byte"},
}


def to_schema(annotation: Any) -> tuple[dict[str, Any], bool]:
    """Convert a Python type annotation to (schema, is_required).

    `is_required` flips to False for Optional[T] / T | None / explicit `None`
    union members. Callers still combine this with default-value presence to
    decide membership in the final `required` list.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}, True
    if annotation is None or annotation is type(None):
        return {"type": "null"}, True

    # `X | Y` (PEP 604) and `Union[X, Y]` go through the same Union handling.
    origin = get_origin(annotation)
    if origin is Union or (UnionType is not None and isinstance(annotation, UnionType)):
        return _schema_for_union(annotation)

    if origin is Literal:
        return _schema_for_literal(annotation)

    if origin in (list, tuple, set, frozenset):
        return _schema_for_sequence(annotation, origin)

    if origin is dict:
        return _schema_for_dict(annotation)

    if isinstance(annotation, type):
        if annotation in _PRIMITIVES:
            return dict(_PRIMITIVES[annotation]), True
        if issubclass(annotation, Enum):
            return _schema_for_enum(annotation), True
        if dataclasses.is_dataclass(annotation):
            return _schema_for_dataclass(annotation), True
        # Bare generics like `list`, `dict` without parameters.
        if annotation is list:
            return {"type": "array"}, True
        if annotation is dict:
            return {"type": "object"}, True
        if annotation is tuple:
            return {"type": "array"}, True
        if annotation is set or annotation is frozenset:
            return {"type": "array", "uniqueItems": True}, True

    # Unknown / unsupported -> empty schema is the JSON Schema "anything" idiom.
    return {}, True


def _schema_for_union(annotation: Any) -> tuple[dict[str, Any], bool]:
    args = [a for a in get_args(annotation) if a is not type(None)]
    has_none = len(args) != len(get_args(annotation))
    if not args:
        return {"type": "null"}, not has_none
    if len(args) == 1:
        inner_schema, _ = to_schema(args[0])
        return inner_schema, not has_none
    inner = [to_schema(a)[0] for a in args]
    return {"anyOf": inner}, not has_none


def _schema_for_literal(annotation: Any) -> tuple[dict[str, Any], bool]:
    values = list(get_args(annotation))
    schema: dict[str, Any] = {"enum": values}
    types = {type(v) for v in values}
    if types == {str}:
        schema["type"] = "string"
    elif types == {int}:
        schema["type"] = "integer"
    elif types <= {int, float}:
        schema["type"] = "number"
    elif types == {bool}:
        schema["type"] = "boolean"
    return schema, True


def _schema_for_sequence(annotation: Any, origin: type) -> tuple[dict[str, Any], bool]:
    args = get_args(annotation)
    base: dict[str, Any] = {"type": "array"}
    if origin in (set, frozenset):
        base["uniqueItems"] = True
    if args:
        # tuple[int, str] -> heterogeneous; we just describe the item union for simplicity.
        if origin is tuple and len(args) > 1 and args[-1] is not Ellipsis:
            item_schemas = [to_schema(a)[0] for a in args]
            base["items"] = {"anyOf": item_schemas} if len(item_schemas) > 1 else item_schemas[0]
        elif origin is tuple and len(args) == 2 and args[-1] is Ellipsis:
            base["items"] = to_schema(args[0])[0]
        else:
            base["items"] = to_schema(args[0])[0]
    return base, True


def _schema_for_dict(annotation: Any) -> tuple[dict[str, Any], bool]:
    args = get_args(annotation)
    base: dict[str, Any] = {"type": "object"}
    if len(args) == 2:
        # We only support string keys, which is JSON Schema's only key type anyway.
        base["additionalProperties"] = to_schema(args[1])[0]
    return base, True


def _schema_for_enum(enum_cls: type[Enum]) -> dict[str, Any]:
    values = [m.value for m in enum_cls]
    schema: dict[str, Any] = {"enum": values}
    types = {type(v) for v in values}
    if types == {str}:
        schema["type"] = "string"
    elif types == {int}:
        schema["type"] = "integer"
    elif types <= {int, float}:
        schema["type"] = "number"
    return schema


def _schema_for_dataclass(cls: type) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        field_schema, field_required = to_schema(f.type if not isinstance(f.type, str) else Any)
        properties[f.name] = field_schema
        if (
            field_required
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            required.append(f.name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema
