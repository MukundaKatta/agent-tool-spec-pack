# agent-tool-spec-pack

[![PyPI](https://img.shields.io/pypi/v/agent-tool-spec-pack.svg)](https://pypi.org/project/agent-tool-spec-pack/)
[![Python](https://img.shields.io/pypi/pyversions/agent-tool-spec-pack.svg)](https://pypi.org/project/agent-tool-spec-pack/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Turn a Python function into a tool-call schema for any major LLM provider.**

You write one Python function with type hints and a docstring. This library
hands you the exact dict that Anthropic's `tools=[...]`, OpenAI's
`tools=[...]`, Gemini's `function_declarations=[...]`, or Bedrock expect. Zero
runtime dependencies, just stdlib `inspect`, `typing`, and `re`.

## Install

```bash
pip install agent-tool-spec-pack
```

## Use

```python
from agent_tool_spec_pack import pack, Format

def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Tokyo" or "San Francisco, CA".
        units: Temperature units. One of "celsius" or "fahrenheit".
    """
    ...

anthropic_spec = pack(get_weather, format=Format.ANTHROPIC)
openai_spec    = pack(get_weather, format=Format.OPENAI)
gemini_spec    = pack(get_weather, format=Format.GEMINI)
```

The three shapes side by side:

```python
# Anthropic
{
  "name": "get_weather",
  "description": "Get the current weather for a city.",
  "input_schema": {
    "type": "object",
    "properties": {
      "city":  {"type": "string", "description": "The city name, ..."},
      "units": {"type": "string", "description": "Temperature units, ...", "default": "celsius"}
    },
    "required": ["city"]
  }
}

# OpenAI
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {...}, "required": ["city"]}
  }
}

# Gemini (camelCase nested keys)
{
  "name": "get_weather",
  "description": "Get the current weather for a city.",
  "parameters": {"type": "object", "properties": {...}, "required": ["city"]}
}
```

Batch many at once:

```python
from agent_tool_spec_pack import pack_many

specs = pack_many([get_weather, search_flights, book_hotel], format=Format.OPENAI)
```

Need to debug what the packer sees before provider shaping?

```python
from agent_tool_spec_pack import pack_inspect

result = pack_inspect(get_weather)
result.name           # "get_weather"
result.description    # "Get the current weather for a city."
result.required       # ["city"]
result.params         # list of ParamInfo with annotation, schema, default, doc
result.skipped        # ["self"], ["*args"], etc.
```

## Supported formats

| Format                     | Shape                                                          |
| -------------------------- | -------------------------------------------------------------- |
| `Format.ANTHROPIC`         | `{name, description, input_schema}`                            |
| `Format.BEDROCK_ANTHROPIC` | identical to `ANTHROPIC` (Bedrock forwards the same shape)     |
| `Format.OPENAI`            | `{type: "function", function: {name, description, parameters}}`|
| `Format.GEMINI`            | `{name, description, parameters}` with camelCase nested keys   |
| `Format.BEDROCK_LLAMA`     | `{name, description, parameters}` (flat, no `function` wrap)   |

## Docstring styles

The packer auto-detects three popular styles. The first one that finds
structured params wins; otherwise the first paragraph becomes the description
and per-param descriptions are left blank.

```python
# Google
"""
Args:
    city: The city name.
    units: celsius or fahrenheit.
"""

# reST / Sphinx
"""
:param city: The city name.
:param units: celsius or fahrenheit.
"""

# numpy
"""
Parameters
----------
city : str
    The city name.
units : str, optional
    celsius or fahrenheit.
"""
```

## Type mapping

| Python annotation                       | JSON Schema fragment                                 |
| --------------------------------------- | ---------------------------------------------------- |
| `int`                                   | `{"type": "integer"}`                                |
| `float`                                 | `{"type": "number"}`                                 |
| `str`                                   | `{"type": "string"}`                                 |
| `bool`                                  | `{"type": "boolean"}`                                |
| `list[T]`                               | `{"type": "array", "items": <schema for T>}`         |
| `dict[str, T]`                          | `{"type": "object", "additionalProperties": <T>}`    |
| `Optional[T]` / `T \| None`             | schema for T, parameter marked not required          |
| `Literal["a", "b"]`                     | `{"type": "string", "enum": ["a", "b"]}`             |
| `Enum` subclass                         | enum of `.value`s                                    |
| `@dataclass` class                      | object with one property per field                   |
| `Union[A, B]`                           | `{"anyOf": [<A>, <B>]}`                              |

Required vs optional: a parameter is required if its signature has no default
value and its type is not Optional. Defaults are copied into the schema's
`default` key.

## What it skips

- `self` and `cls` (implicit instance / class arguments).
- `*args` and `**kwargs` (LLM tool schemas can't express them). A warning fires
  so you notice if you forgot to convert them to named params.

## Pairs nicely with

- [`agentvet`](https://pypi.org/project/agentvet/) - validate tool arguments before execution.
- [`llm-tool-arg-coerce`](https://pypi.org/project/llm-tool-arg-coerce/) - coerce LLM string output back to Python types.
- [`agent-tool-graph`](https://pypi.org/project/agent-tool-graph/) - run tools in prerequisite order.

`agent-tool-spec-pack` runs first (publish the schemas to the LLM), then
`llm-tool-arg-coerce` and `agentvet` run on the way back in, then
`agent-tool-graph` decides what to run.

## License

MIT
