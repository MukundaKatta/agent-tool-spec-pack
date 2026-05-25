# agent-tool-spec-pack

Convert Python function signatures to Anthropic / OpenAI / Gemini tool schemas.

```python
from agent_tool_spec_pack import ToolSpecPack

pack = ToolSpecPack()

@pack.register
def search_web(query: str, limit: int = 10) -> list:
    """Search the web and return results."""
    ...

@pack.register(description="Send an email")
def send_email(to: str, subject: str, body: str) -> bool:
    ...

# export to any provider
anthropic_tools = pack.to_anthropic()
openai_tools    = pack.to_openai()
gemini_tools    = pack.to_gemini()
```

Zero dependencies. Type annotations map automatically to JSON Schema types.
