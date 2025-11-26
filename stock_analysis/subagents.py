from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence


from .utils import read_prompt


def build_subagents(
    prompts_root: Path,
    mcp_tools: Sequence[Any],
    av_tools: Sequence[Any],
    web_tools: Sequence[Any],
    chart_tools: Sequence[Any],
) -> List[Dict[str, Any]]:
    _ = mcp_tools  # Placeholder for future MCP-specific tool routing.
    def P(name: str) -> str:
        # Subagents do NOT receive global principles here.
        # The main agent will pass relevant principles per task in the task input context.
        return read_prompt(prompts_root / name)

    chart_guidance = (
        "\n\nVisualization guidance:\n"
        "- You can call the shared chart tools (bar/line/pie) whenever visuals help.\n"
        "- For complex diagrams or flows, emit Mermaid syntax inside ```mermaid code fences.\n"
        "- When embedding generated charts, always use the absolute path returned by the tool (do not rewrite it to a relative path)."
    )

    def _merge_with_chart_tools(existing_tools: Sequence[Any]) -> List[Any]:
        merged: List[Any] = list(existing_tools)
        seen = {
            getattr(tool, "name", "") or f"id:{id(tool)}"
            for tool in merged
        }
        for tool in chart_tools:
            key = getattr(tool, "name", "") or f"id:{id(tool)}"
            if key in seen:
                continue
            merged.append(tool)
            seen.add(key)
        return merged

    base_subagents = [
        {
            "name": "supervisor",
            "description": "Coordinator that plans, gates, and orchestrates research.",
            "system_prompt": P("supervisor.txt"),
            # Only FS/Todo (from built-in middleware); no MCP tools here.
            "tools": [],
        },
        {
            "name": "websearch",
            "description": "General web search for broad context and reputable sources; deduplicate and summarize.",
            "system_prompt": P("websearch.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "fundamentals",
            "description": "Fetch and normalize fundamentals from Alpha Vantage.",
            "system_prompt": P("fundamentals.txt"),
            "tools": list(av_tools),
        },
        {
            "name": "prices",
            "description": "Get price time series, compute derived P/B, and produce chart artifacts.",
            "system_prompt": P("prices.txt"),
            "tools": list(av_tools) + list(chart_tools),
        },
        {
            "name": "filings_ownership_legal",
            "description": "Filings, legal/governance, and ownership mapping.",
            "system_prompt": P("filings_ownership_legal.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "divisions",
            "description": "Business divisions, product/geography mix, and trends.",
            "system_prompt": P("divisions.txt"),
            "tools": list(av_tools) + list(web_tools),
        },
        {
            "name": "cashflow",
            "description": "Compute FCF breakdown and acquisitions vs cashflow tables.",
            "system_prompt": P("cashflow.txt"),
            "tools": list(av_tools) + list(web_tools),
        },
        {
            "name": "cashpile",
            "description": "Track cash/financial assets and returns.",
            "system_prompt": P("cashpile.txt"),
            "tools": list(av_tools) + list(web_tools),
        },
        {
            "name": "risk",
            "description": "Identify and document key risks with citations.",
            "system_prompt": P("risk.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "family",
            "description": "Build family tree and share distribution notes.",
            "system_prompt": P("family.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "management",
            "description": "Summarize key managers and involvement/side activities.",
            "system_prompt": P("management.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "board",
            "description": "Summarize board ages and status.",
            "system_prompt": P("board.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "historical_trading",
            "description": "Compile trading history (who/when/amount/holding).",
            "system_prompt": P("historical_trading.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "asset_notes",
            "description": "Summarize asset-specific notes.",
            "system_prompt": P("asset_notes.txt"),
            "tools": list(web_tools),
        },
        {
            "name": "valuation",
            "description": "Build SOP, liquidation, buyout/rights tables and IRRs.",
            "system_prompt": P("valuation.txt"),
            # Needs only FS access (provided by middleware), no MCP tools.
            "tools": [],
        },
        {
            "name": "writer",
            "description": "Assemble final report from artifacts and template.",
            "system_prompt": P("writer.txt"),
            # Needs only FS access (provided by middleware) plus charting tools.
            "tools": list(chart_tools),
        },
    ]

    for agent in base_subagents:
        agent["tools"] = _merge_with_chart_tools(agent.get("tools", []))
        agent["system_prompt"] = f"{agent['system_prompt']}{chart_guidance}"

    return base_subagents
