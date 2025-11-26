import pytest

from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException

from stock_analysis.utils import (
    filter_non_tavily_tools,
    filter_tavily_tools,
    filter_out_tools_by_names,
    filter_out_alpha_vantage_commodities,
    wrap_tools_with_error_handler,
    wrap_tools_with_extract_materializer,
)


class DummyTool:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.handle_tool_error = None


def test_filter_helpers_separate_tavily_tools():
    tools = [
        DummyTool("tavily_search", "Tavily search the web"),
        DummyTool("alpha_fundamentals", "Alpha Vantage fundamentals"),
    ]

    tavily = filter_tavily_tools(tools)
    non_tavily = filter_non_tavily_tools(tools)

    assert [t.name for t in tavily] == ["tavily_search"]
    assert [t.name for t in non_tavily] == ["alpha_fundamentals"]


def test_filter_out_tools_by_names_discards_matches():
    tools = [
        DummyTool("tavily_map"),
        DummyTool("tavily_search"),
        DummyTool("alpha_fundamentals"),
    ]

    remaining = filter_out_tools_by_names(tools, names={"tavily_map", "unused"})

    assert [t.name for t in remaining] == ["tavily_search", "alpha_fundamentals"]


def test_filter_out_alpha_vantage_commodities_removes_known_tools():
    tools = [
        DummyTool("NATURAL_GAS"),
        DummyTool("alpha_fundamentals"),
        DummyTool("copper"),
        DummyTool("custom_tool"),
    ]

    remaining = filter_out_alpha_vantage_commodities(tools)

    assert [t.name for t in remaining] == ["alpha_fundamentals", "custom_tool"]


def test_wrap_tools_with_error_handler_captures_validation_text():
    tavily_tool = DummyTool("tavily_search")
    other_tool = DummyTool("alpha_data")

    wrapped = wrap_tools_with_error_handler([tavily_tool, other_tool])

    handler = wrapped[0].handle_tool_error
    assert callable(handler)

    message = handler(ToolException("Validation error: missing field"))
    assert "missing field" in message

    with pytest.raises(ToolException):
        handler(ToolException("unexpected failure"))

    assert wrapped[1].handle_tool_error is None


def test_wrap_tools_with_extract_materializer_persists_payload(tmp_path):
    def fake_extract(url: str):
        return {"url": url, "content": "Example extracted text"}

    extract_tool = StructuredTool.from_function(
        name="tavily_extract",
        description="Tavily extract large payload",
        func=fake_extract,
    )
    passthrough_tool = StructuredTool.from_function(
        name="alpha_data_lookup",
        description="Alpha lookup",
        func=lambda symbol: {"symbol": symbol},
    )

    wrapped = wrap_tools_with_extract_materializer(
        [extract_tool, passthrough_tool],
        workspace_dir=tmp_path,
        max_chars=20,
    )

    materialized = wrapped[0]
    result = materialized.func(url="https://example.com/page")

    assert result["items"][0]["chunks"] >= 1
    saved_paths = result["saved"]
    assert saved_paths
    for rel_path in saved_paths:
        assert (tmp_path / rel_path).exists()

    # Ensure non-matching tools are passed through unchanged.
    assert wrapped[1] is passthrough_tool

