"""
agent-tool-spec-pack: Convert Python function signatures to multi-provider tool schemas.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints


_PYTHON_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    bytes: "string",
}


def _python_type_to_json(t: Any) -> str:
    if t is inspect.Parameter.empty or t is None:
        return "string"
    origin = getattr(t, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    return _PYTHON_TYPE_MAP.get(t, "string")


@dataclass
class ParameterSpec:
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = inspect.Parameter.empty
    enum: Optional[list[Any]] = None
    items: Optional[dict[str, Any]] = None

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.items is not None and self.type == "array":
            schema["items"] = self.items
        return schema


@dataclass
class ToolSpec:
    """Provider-agnostic tool specification."""
    name: str
    description: str
    parameters: list[ParameterSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def required_params(self) -> list[str]:
        return [p.name for p in self.parameters if p.required]

    def to_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {p.name: p.to_json_schema() for p in self.parameters},
            "required": self.required_params,
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.to_json_schema(),
        }

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }

    def to_gemini(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.to_json_schema(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {"name": p.name, "type": p.type, "description": p.description, "required": p.required}
                for p in self.parameters
            ],
        }


def spec_from_fn(
    fn: Callable[..., Any],
    description: str = "",
    param_descriptions: Optional[dict[str, str]] = None,
    skip_params: Optional[list[str]] = None,
) -> ToolSpec:
    """Build a ToolSpec from a Python function's signature."""
    skip = set(skip_params or [])
    desc = description or (inspect.getdoc(fn) or "").split("\n")[0].strip()
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    sig = inspect.signature(fn)
    params: list[ParameterSpec] = []
    for name, param in sig.parameters.items():
        if name in skip:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(name, inspect.Parameter.empty)
        json_type = _python_type_to_json(ann)
        has_default = param.default is not inspect.Parameter.empty
        param_desc = (param_descriptions or {}).get(name, "")
        params.append(ParameterSpec(
            name=name,
            type=json_type,
            description=param_desc,
            required=not has_default,
            default=param.default,
        ))
    return ToolSpec(name=fn.__name__, description=desc, parameters=params)


class ToolSpecPack:
    """Registry of ToolSpecs with bulk export to any provider format."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def add(self, spec: ToolSpec) -> "ToolSpecPack":
        self._specs[spec.name] = spec
        return self

    def register(
        self,
        fn: Optional[Callable[..., Any]] = None,
        *,
        description: str = "",
        param_descriptions: Optional[dict[str, str]] = None,
        skip_params: Optional[list[str]] = None,
    ) -> Any:
        """Decorator: @pack.register or @pack.register(description='...')."""
        def _decorate(f: Callable[..., Any]) -> Callable[..., Any]:
            spec = spec_from_fn(f, description=description,
                                param_descriptions=param_descriptions,
                                skip_params=skip_params)
            self._specs[spec.name] = spec
            return f
        if fn is not None:
            return _decorate(fn)
        return _decorate

    def get(self, name: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(f"No tool named {name!r}")
        return self._specs[name]

    def get_or_none(self, name: str) -> Optional[ToolSpec]:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return list(self._specs.keys())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def to_anthropic(self) -> list[dict[str, Any]]:
        return [s.to_anthropic() for s in self._specs.values()]

    def to_openai(self) -> list[dict[str, Any]]:
        return [s.to_openai() for s in self._specs.values()]

    def to_gemini(self) -> list[dict[str, Any]]:
        return [s.to_gemini() for s in self._specs.values()]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_anthropic(), indent=indent)


__all__ = ["ToolSpecPack", "ToolSpec", "ParameterSpec", "spec_from_fn"]
