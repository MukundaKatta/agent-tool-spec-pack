"""Tests for agent-tool-spec-pack."""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Union

from agent_tool_spec_pack import Format, pack, pack_inspect, pack_many

# ---------------------------------------------------------------------------
# Fixtures: callables shaped for each parser
# ---------------------------------------------------------------------------


def google_doc_fn(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name, e.g. "Tokyo" or "San Francisco, CA".
        units: Temperature units. One of "celsius" or "fahrenheit".
    """
    return ""


def rest_doc_fn(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    :param city: The city name. Can be a place like "Tokyo".
    :param units: Temperature units, celsius or fahrenheit.
    :returns: A weather string.
    """
    return ""


def numpy_doc_fn(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Parameters
    ----------
    city : str
        The city name.
    units : str, optional
        Temperature units, celsius or fahrenheit.

    Returns
    -------
    str
        A weather string.
    """
    return ""


def no_args_section_fn(city: str) -> str:
    """Get the current weather for a city.

    This second paragraph should not be part of the description.
    """
    return ""


# ---------------------------------------------------------------------------
# Provider shape tests
# ---------------------------------------------------------------------------


def test_anthropic_shape_has_input_schema():
    spec = pack(google_doc_fn, format=Format.ANTHROPIC)
    assert spec["name"] == "google_doc_fn"
    assert spec["description"] == "Get the current weather for a city."
    assert spec["input_schema"]["type"] == "object"
    assert "city" in spec["input_schema"]["properties"]
    assert spec["input_schema"]["required"] == ["city"]


def test_openai_shape_wraps_function_field():
    spec = pack(google_doc_fn, format=Format.OPENAI)
    assert spec["type"] == "function"
    inner = spec["function"]
    assert inner["name"] == "google_doc_fn"
    assert inner["parameters"]["type"] == "object"
    assert inner["parameters"]["required"] == ["city"]


def test_gemini_shape_uses_parameters_key():
    spec = pack(google_doc_fn, format=Format.GEMINI)
    assert spec["name"] == "google_doc_fn"
    assert "parameters" in spec
    assert "input_schema" not in spec
    assert spec["parameters"]["type"] == "object"


def test_bedrock_anthropic_matches_anthropic():
    a = pack(google_doc_fn, format=Format.ANTHROPIC)
    b = pack(google_doc_fn, format=Format.BEDROCK_ANTHROPIC)
    assert a == b


def test_bedrock_llama_uses_parameters_flat():
    spec = pack(google_doc_fn, format=Format.BEDROCK_LLAMA)
    assert spec["name"] == "google_doc_fn"
    assert "parameters" in spec
    assert "function" not in spec  # not nested like OpenAI
    assert "input_schema" not in spec  # not Anthropic-shaped


# ---------------------------------------------------------------------------
# Docstring parsing tests
# ---------------------------------------------------------------------------


def test_google_style_param_descriptions():
    spec = pack(google_doc_fn, format=Format.ANTHROPIC)
    props = spec["input_schema"]["properties"]
    assert "Tokyo" in props["city"]["description"]
    assert "celsius" in props["units"]["description"]


def test_rest_style_param_descriptions():
    spec = pack(rest_doc_fn, format=Format.ANTHROPIC)
    props = spec["input_schema"]["properties"]
    assert "Tokyo" in props["city"]["description"]
    assert "celsius" in props["units"]["description"]


def test_numpy_style_param_descriptions():
    spec = pack(numpy_doc_fn, format=Format.ANTHROPIC)
    props = spec["input_schema"]["properties"]
    assert "city name" in props["city"]["description"].lower()
    assert "celsius" in props["units"]["description"].lower()


def test_first_paragraph_fallback_when_no_args_section():
    spec = pack(no_args_section_fn, format=Format.ANTHROPIC)
    assert spec["description"] == "Get the current weather for a city."
    # No per-param description was found, so the property dict has no description key.
    assert "description" not in spec["input_schema"]["properties"]["city"]


def test_missing_docstring_yields_empty_description():
    def bare(city: str) -> str:
        return ""

    spec = pack(bare, format=Format.ANTHROPIC)
    assert spec["description"] == ""


# ---------------------------------------------------------------------------
# Type mapping tests (one per row in the README table)
# ---------------------------------------------------------------------------


def _schema_for(fn) -> dict:
    return pack(fn, format=Format.ANTHROPIC)["input_schema"]["properties"]


def test_type_int_maps_to_integer():
    def fn(x: int) -> None: ...

    assert _schema_for(fn)["x"] == {"type": "integer"}


def test_type_float_maps_to_number():
    def fn(x: float) -> None: ...

    assert _schema_for(fn)["x"] == {"type": "number"}


def test_type_str_maps_to_string():
    def fn(x: str) -> None: ...

    assert _schema_for(fn)["x"] == {"type": "string"}


def test_type_bool_maps_to_boolean():
    def fn(x: bool) -> None: ...

    assert _schema_for(fn)["x"] == {"type": "boolean"}


def test_type_list_with_inner_type():
    def fn(x: list[int]) -> None: ...

    assert _schema_for(fn)["x"] == {"type": "array", "items": {"type": "integer"}}


def test_type_dict_with_value_type():
    def fn(x: dict[str, float]) -> None: ...

    assert _schema_for(fn)["x"] == {
        "type": "object",
        "additionalProperties": {"type": "number"},
    }


def test_optional_param_is_not_required():
    def fn(x: int, y: Optional[str] = None) -> None: ...  # noqa: UP007, UP045

    spec = pack(fn, format=Format.ANTHROPIC)
    assert spec["input_schema"]["required"] == ["x"]


def test_pep604_optional_is_not_required():
    def fn(x: int, y: str | None = None) -> None: ...

    spec = pack(fn, format=Format.ANTHROPIC)
    assert spec["input_schema"]["required"] == ["x"]


def test_literal_becomes_enum_with_string_type():
    def fn(units: Literal["celsius", "fahrenheit"]) -> None: ...

    schema = _schema_for(fn)["units"]
    assert schema["enum"] == ["celsius", "fahrenheit"]
    assert schema["type"] == "string"


def test_enum_subclass_uses_value_list():
    class Color(str, Enum):
        RED = "red"
        BLUE = "blue"

    def fn(c: Color) -> None: ...

    schema = _schema_for(fn)["c"]
    assert schema["enum"] == ["red", "blue"]
    assert schema["type"] == "string"


def test_dataclass_becomes_object_with_properties():
    @dataclass
    class Point:
        x: int
        y: int
        label: str = "origin"

    def fn(p: Point) -> None: ...

    schema = _schema_for(fn)["p"]
    assert schema["type"] == "object"
    assert set(schema["properties"]) == {"x", "y", "label"}
    assert schema["required"] == ["x", "y"]


# ---------------------------------------------------------------------------
# Required / default-handling tests
# ---------------------------------------------------------------------------


def test_required_when_no_default():
    def fn(must_have: int) -> None: ...

    spec = pack(fn, format=Format.ANTHROPIC)
    assert spec["input_schema"]["required"] == ["must_have"]


def test_optional_when_default_present():
    def fn(can_skip: int = 5) -> None: ...

    spec = pack(fn, format=Format.ANTHROPIC)
    assert spec["input_schema"]["required"] == []
    assert spec["input_schema"]["properties"]["can_skip"]["default"] == 5


def test_default_value_round_trips_basic_primitives():
    def fn(a: int = 1, b: str = "hi", c: bool = True) -> None: ...

    props = _schema_for(fn)
    assert props["a"]["default"] == 1
    assert props["b"]["default"] == "hi"
    assert props["c"]["default"] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_self_is_skipped():
    class Tool:
        def call(self, query: str) -> str:
            """Tool entrypoint.

            Args:
                query: The user query.
            """
            return ""

    spec = pack(Tool().call, format=Format.ANTHROPIC)
    assert "self" not in spec["input_schema"]["properties"]
    assert list(spec["input_schema"]["properties"]) == ["query"]


def test_args_kwargs_emit_warning_and_are_skipped():
    def fn(query: str, *args: int, **kwargs: str) -> None:
        """Args:
        query: The query.
        """

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spec = pack(fn, format=Format.ANTHROPIC)
    assert any("args" in str(wi.message) or "kwargs" in str(wi.message) for wi in w)
    assert list(spec["input_schema"]["properties"]) == ["query"]


def test_gemini_camel_case_normalizes_dict_value_type():
    def fn(x: dict[str, int]) -> None: ...

    spec = pack(fn, format=Format.GEMINI)
    props = spec["parameters"]["properties"]
    # `additional_properties`-ish keys get camelCased for Gemini.
    assert "additionalProperties" in props["x"]
    assert props["x"]["additionalProperties"] == {"type": "integer"}


def test_pack_many_returns_list_in_order():
    def a(x: int) -> None: ...
    def b(y: str) -> None: ...

    specs = pack_many([a, b], format=Format.ANTHROPIC)
    assert isinstance(specs, list)
    assert [s["name"] for s in specs] == ["a", "b"]


def test_pack_inspect_exposes_intermediate_state():
    result = pack_inspect(google_doc_fn)
    assert result.name == "google_doc_fn"
    assert result.required == ["city"]
    names = [p.name for p in result.params]
    assert names == ["city", "units"]
    assert any("Tokyo" in p.description for p in result.params)


def test_union_with_multiple_real_types_uses_any_of():
    def fn(x: Union[int, str]) -> None: ...  # noqa: UP007

    schema = _schema_for(fn)["x"]
    assert "anyOf" in schema
    assert {"type": "integer"} in schema["anyOf"]
    assert {"type": "string"} in schema["anyOf"]


def test_format_accepts_string_alias():
    spec_from_enum = pack(google_doc_fn, format=Format.OPENAI)
    spec_from_str = pack(google_doc_fn, format="openai")
    assert spec_from_enum == spec_from_str


def test_dataclass_with_default_factory_is_optional():
    @dataclass
    class Bag:
        items: list[str] = field(default_factory=list)

    def fn(b: Bag) -> None: ...

    schema = _schema_for(fn)["b"]
    # No "required" key when every field is optional.
    assert "required" not in schema
