from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime, UTC

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
# from playwright.async_api import async_playwright
# from langchain_tavily import TavilySearch  # Direct SDK; currently using Tavily MCP tools instead

from . import config
from .utils import (
    prompts_dir,
    templates_dir,
    read_prompt,
    inject_principles,
    filter_non_tavily_tools,
    filter_tavily_tools,
    filter_out_tools_by_names,
    wrap_tools_with_error_handler,
    wrap_tools_with_extract_materializer,
)
from .subagents import build_subagents


async def build_agent(principles: Optional[str] = None) -> Any:
    # Collect MCP tools (dict config with per-server transport)
    mcp_client = MultiServerMCPClient(config.MCP_SERVERS)
    mcp_tools = await mcp_client.get_tools()
    # For now, treat "Alpha Vantage tools" as all non-Tavily tools
    av_tools = filter_non_tavily_tools(mcp_tools)
    # Initialize Playwright browser tools (async API) compatible with running event loop
    # pw = await async_playwright().start()
    # browser = await pw.chromium.launch(headless=True)
    # pw_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    # web_tools = pw_toolkit.get_tools()
    # Initialize Tavily search tool (replace Playwright/browser MCP for websearch) via SDK
    # tavily_tool = TavilySearch(max_results=5)
    # web_tools = [tavily_tool]
    # Use Tavily MCP tools for web search
    web_tools = filter_tavily_tools(mcp_tools)
    # Exclude specific Tavily tools (e.g., map) that we don't want subagents to call
    web_tools = filter_out_tools_by_names(web_tools, names={"tavily_map"})
    # Wrap extract materializer to the web_tools
    web_tools = wrap_tools_with_extract_materializer(web_tools, config.WORKSPACE_DIR)
    # Make tool errors non-fatal so the model can self-correct
    web_tools = wrap_tools_with_error_handler(web_tools)
    # If you want to fall back to Browser MCP tools as well, use this instead:
    # web_tools = filter_tavily_tools(mcp_tools) or filter_browser_tools(mcp_tools)

    # Ensure scratchpad directory/files
    scratch_dir: Path = config.WORKSPACE_DIR / "scratchpad"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    global_scratchpad = scratch_dir / "global_scratchpad.md"
    if not global_scratchpad.exists():
        ts = datetime.now(UTC).isoformat(timespec="seconds") + "Z"
        global_scratchpad.write_text(
            f"# Global Scratchpad\n\nInitialized at {ts}\n", encoding="utf-8"
        )

    # Load prompts
    prompts_root = prompts_dir()
    system_prompt = inject_principles(
        read_prompt(prompts_root / "system.txt"), principles
    )
    # Include the report template so the main agent can format output consistently
    try:
        template_text = read_prompt(templates_dir() / "report_template.md")
        system_prompt = f"{system_prompt}\n\nReport format template:\n{template_text}"
    except (OSError, FileNotFoundError, UnicodeDecodeError):
        # If template missing, proceed without blocking
        pass

    # Subagents (context-isolated specialists)
    subagents: List[Dict[str, Any]] = build_subagents(
        prompts_root=prompts_root,
        mcp_tools=mcp_tools,
        av_tools=av_tools,
        web_tools=web_tools,
    )

    # Append scratchpad guidance to main and subagent prompts, and ensure per-agent files
    scratch_instructions_main = (
        "\n\nScratchpad policy:\n"
        "- Log every step of progress to the global scratchpad: scratchpad/global_scratchpad.md\n"
        "- Before each step, read the global scratchpad for prior progress.\n"
        "- Use concise bullet points with ISO 8601 UTC timestamps.\n"
    )
    system_prompt = f"{system_prompt}{scratch_instructions_main}"

    augmented_subagents: List[Dict[str, Any]] = []
    for sa in subagents:
        name = sa.get("name", "agent")
        agent_pad = scratch_dir / f"{name}_scratchpad.md"
        if not agent_pad.exists():
            ts = datetime.now(UTC).isoformat(timespec="seconds") + "Z"
            agent_pad.write_text(
                f"# {name} Scratchpad\n\nInitialized at {ts}\n", encoding="utf-8"
            )
        scratch_instructions = (
            "\n\nScratchpad policy:\n"
            f"- Your scratchpad: scratchpad/{name}_scratchpad.md\n"
            "- Also append a brief step header to scratchpad/global_scratchpad.md\n"
            "- Before each step, read both scratchpads and continue from the last checkpoint.\n"
            "- Use concise bullet points with ISO 8601 UTC timestamps.\n"
        )
        sa_aug = dict(sa)
        sa_aug["system_prompt"] = f"{sa.get('system_prompt', '')}{scratch_instructions}"
        augmented_subagents.append(sa_aug)

    # Pass no MCP tools to the main agent; use FilesystemBackend to store in workspace directory.
    agent = create_deep_agent(
        model=config.MODEL,
        tools=[],
        system_prompt=system_prompt,
        subagents=augmented_subagents,
        backend=FilesystemBackend(
            root_dir=str(config.WORKSPACE_DIR), virtual_mode=True
        ),
    )
    return agent
