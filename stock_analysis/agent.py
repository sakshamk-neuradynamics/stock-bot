from __future__ import annotations

from typing import Any, Dict, List, Optional

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_mcp_adapters.client import MultiServerMCPClient
# from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
# from playwright.async_api import async_playwright
# from langchain_tavily import TavilySearch  # Direct SDK; currently using Tavily MCP tools instead

from . import config
from .utils import prompts_dir, templates_dir, read_prompt, inject_principles, filter_alpha_vantage_tools, filter_tavily_tools, wrap_tools_with_error_handler
from .subagents import build_subagents

async def build_agent(principles: Optional[str] = None) -> Any:
    # Collect MCP tools (dict config with per-server transport)
    mcp_client = MultiServerMCPClient(config.MCP_SERVERS)
    mcp_tools = await mcp_client.get_tools()
    av_tools = filter_alpha_vantage_tools(mcp_tools)
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
    # Make tool errors non-fatal so the model can self-correct
    web_tools = wrap_tools_with_error_handler(web_tools)
    # If you want to fall back to Browser MCP tools as well, use this instead:
    # web_tools = filter_tavily_tools(mcp_tools) or filter_browser_tools(mcp_tools)

    # Load prompts
    prompts_root = prompts_dir()
    system_prompt = inject_principles(read_prompt(prompts_root / "system.txt"), principles)
    # Include the report template so the main agent can format output consistently
    try:
        template_text = read_prompt(templates_dir() / "report_template.md")
        system_prompt = f"{system_prompt}\n\nReport format template:\n{template_text}"
    except Exception:
        # If template missing, proceed without blocking
        pass

    # Subagents (context-isolated specialists)
    subagents: List[Dict[str, Any]] = build_subagents(
        prompts_root=prompts_root,
        mcp_tools=mcp_tools,
        av_tools=av_tools,
        web_tools=web_tools,
    )

    # Pass no MCP tools to the main agent; use FilesystemBackend to store in workspace directory.
    agent = create_deep_agent(
        model=config.MODEL,
        tools=[],
        system_prompt=system_prompt,
        subagents=subagents,
        backend=FilesystemBackend(root_dir=str(config.WORKSPACE_DIR), virtual_mode=True)
    )
    return agent
