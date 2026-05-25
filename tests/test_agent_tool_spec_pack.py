"""Tests for agent-tool-spec-pack."""
import pytest
from agent_tool_spec_pack import ToolSpecPack, ToolSpec, ParameterSpec, spec_from_fn


def search_web(query: str, limit: int = 10) -> list:
    """Search the web and return results."""
    return []


def no_args() -> str:
    """Returns a string."""
    return "ok"


def test_spec_from_fn_name():
    spec = spec_from_fn(search_web)
    assert spec.name == "search_web"


def test_spec_from_fn_description():
    spec = spec_from_fn(search_web)
    assert "Search" in spec.description


def test_spec_from_fn_required_param():
    spec = spec_from_fn(search_web)
    required = [p for p in spec.parameters if p.name == "query"]
    assert len(required) == 1
    assert required[0].required is True
    assert required[0].type == "string"


def test_spec_from_fn_optional_param():
    spec = spec_from_fn(search_web)
    optional = [p for p in spec.parameters if p.name == "limit"]
    assert len(optional) == 1
    assert optional[0].required is False
    assert optional[0].type == "integer"


def test_spec_from_fn_no_args():
    spec = spec_from_fn(no_args)
    assert spec.parameters == []


def test_spec_from_fn_param_descriptions():
    spec = spec_from_fn(search_web, param_descriptions={"query": "The search query"})
    q = next(p for p in spec.parameters if p.name == "query")
    assert q.description == "The search query"


def test_spec_from_fn_skip_params():
    def fn(self, x: str) -> None:
        pass
    spec = spec_from_fn(fn, skip_params=["self"])
    assert all(p.name != "self" for p in spec.parameters)


def test_tool_spec_to_anthropic():
    spec = spec_from_fn(search_web)
    ant = spec.to_anthropic()
    assert ant["name"] == "search_web"
    assert "input_schema" in ant
    assert ant["input_schema"]["type"] == "object"
    assert "query" in ant["input_schema"]["properties"]


def test_tool_spec_to_openai():
    spec = spec_from_fn(search_web)
    oai = spec.to_openai()
    assert oai["type"] == "function"
    assert oai["function"]["name"] == "search_web"
    assert "parameters" in oai["function"]


def test_tool_spec_to_gemini():
    spec = spec_from_fn(search_web)
    gem = spec.to_gemini()
    assert gem["name"] == "search_web"
    assert "parameters" in gem


def test_tool_spec_required_params():
    spec = spec_from_fn(search_web)
    assert "query" in spec.required_params
    assert "limit" not in spec.required_params


def test_pack_register_decorator():
    pack = ToolSpecPack()

    @pack.register
    def do_thing(x: str) -> None:
        """Does a thing."""

    assert "do_thing" in pack


def test_pack_register_with_args():
    pack = ToolSpecPack()

    @pack.register(description="Custom desc")
    def do_other(x: str) -> None:
        pass

    assert pack.get("do_other").description == "Custom desc"


def test_pack_add():
    pack = ToolSpecPack()
    spec = spec_from_fn(search_web)
    pack.add(spec)
    assert "search_web" in pack


def test_pack_len():
    pack = ToolSpecPack()
    pack.add(spec_from_fn(search_web))
    pack.add(spec_from_fn(no_args))
    assert len(pack) == 2


def test_pack_names():
    pack = ToolSpecPack()
    pack.add(spec_from_fn(search_web))
    assert "search_web" in pack.names()


def test_pack_get_missing():
    pack = ToolSpecPack()
    with pytest.raises(KeyError):
        pack.get("nonexistent")


def test_pack_get_or_none():
    pack = ToolSpecPack()
    assert pack.get_or_none("missing") is None


def test_pack_to_anthropic():
    pack = ToolSpecPack()
    pack.add(spec_from_fn(search_web))
    schemas = pack.to_anthropic()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "search_web"


def test_pack_to_openai():
    pack = ToolSpecPack()
    pack.add(spec_from_fn(search_web))
    schemas = pack.to_openai()
    assert schemas[0]["type"] == "function"


def test_pack_to_gemini():
    pack = ToolSpecPack()
    pack.add(spec_from_fn(search_web))
    schemas = pack.to_gemini()
    assert "parameters" in schemas[0]


def test_pack_to_json():
    pack = ToolSpecPack()
    pack.add(spec_from_fn(search_web))
    j = pack.to_json()
    assert "search_web" in j


def test_parameter_spec_to_schema():
    p = ParameterSpec(name="x", type="string", description="A value", required=True)
    schema = p.to_json_schema()
    assert schema["type"] == "string"
    assert schema["description"] == "A value"


def test_parameter_spec_enum():
    p = ParameterSpec(name="color", type="string", enum=["red", "green", "blue"], required=True)
    schema = p.to_json_schema()
    assert schema["enum"] == ["red", "green", "blue"]
