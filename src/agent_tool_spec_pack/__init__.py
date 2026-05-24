"""agent-tool-spec-pack - convert Python callables into LLM tool-call schemas.

One function, many provider shapes. Hand it a Python callable with type hints
and a docstring; get back the dict that Anthropic, OpenAI, Gemini, or Bedrock
expects in their tool/function-calling payloads.

    from agent_tool_spec_pack import pack, Format

    def get_weather(city: str, units: str = "celsius") -> str:
        '''Get the current weather for a city.

        Args:
            city: The city name.
            units: Temperature units. One of "celsius" or "fahrenheit".
        '''
        ...

    anthropic_spec = pack(get_weather, format=Format.ANTHROPIC)
    openai_spec    = pack(get_weather, format=Format.OPENAI)
    gemini_spec    = pack(get_weather, format=Format.GEMINI)

The packer reads:
- type hints (int / float / str / bool / list / dict / Optional / Literal /
  Enum / dataclass) to derive JSON Schema types
- docstrings in Google, reST, or numpy style for per-parameter descriptions

Pairs with:
- `agentvet` - validate tool args before execution
- `llm-tool-arg-coerce` - coerce LLM string output back to Python types
- `agent-tool-graph` - run tools in prerequisite order
"""

from agent_tool_spec_pack.pack import pack, pack_inspect, pack_many
from agent_tool_spec_pack.types import Format, InspectionResult, ParamInfo

__version__ = "0.1.0"

__all__ = [
    "Format",
    "InspectionResult",
    "ParamInfo",
    "__version__",
    "pack",
    "pack_inspect",
    "pack_many",
]
