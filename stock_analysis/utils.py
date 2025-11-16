from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools.base import ToolException
# (no pydantic imports needed here)




VALUE_PRINCIPLES_TOKEN = "{{VALUE_INVESTING_PRINCIPLES}}"


def prompts_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"

def templates_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def inject_principles(text: str, principles: Optional[str]) -> str:
    if principles and VALUE_PRINCIPLES_TOKEN in text:
        return text.replace(VALUE_PRINCIPLES_TOKEN, principles)
    return text


def _name_or_desc(tool: Any) -> str:
    name = getattr(tool, "name", "") or ""
    desc = getattr(tool, "description", "") or ""
    return f"{name} {desc}".lower()


def filter_alpha_vantage_tools(tools: Iterable[Any]) -> List[Any]:
    results: List[Any] = []
    for t in tools:
        text = _name_or_desc(t)
        if "alpha" in text or "vantag" in text or "alphavantage" in text:
            results.append(t)
    return results


def filter_browser_tools(tools: Iterable[Any]) -> List[Any]:
    results: List[Any] = []
    browser_keywords = ("browser", "search", "open_url", "get_text", "download")
    for t in tools:
        text = _name_or_desc(t)
        if any(k in text for k in browser_keywords):
            results.append(t)
    return results


def filter_tavily_tools(tools: Iterable[Any]) -> List[Any]:
    results: List[Any] = []
    for t in tools:
        text = _name_or_desc(t)
        if "tavily" in text or "tavily_search" in text:
            results.append(t)
    return results

def filter_non_tavily_tools(tools: Iterable[Any]) -> List[Any]:
    """Return all tools that are not Tavily tools."""
    tavily = filter_tavily_tools(tools)
    tavily_ids = {id(t) for t in tavily}
    return [t for t in tools if id(t) not in tavily_ids]

def filter_out_tools_by_names(tools: Iterable[Any], names: Iterable[str]) -> List[Any]:
    """Exclude tools whose name matches any in 'names' (case-insensitive)."""
    name_set = {n.lower() for n in names}
    results: List[Any] = []
    for t in tools:
        tname = (getattr(t, "name", "") or "").lower()
        if tname not in name_set:
            results.append(t)
    return results


def wrap_tools_with_error_handler(tools: Iterable[Any]) -> List[Any]:
    """Attach a validation-only error handler to Tavily tools; return exact error string; raise others."""
    def _looks_like_validation(err: Exception) -> bool:
        s = str(err).lower()
        return ("validation error" in s) or ("unexpected keyword argument" in s) or ("not one of" in s)

    wrapped: List[Any] = []
    for t in tools:
        name = getattr(t, "name", "") or ""
        if "tavily" in name:
            def _handler(e: Exception) -> str:
                if isinstance(e, ToolException) and _looks_like_validation(e):
                    return str(e)
                raise e
            try:
                setattr(t, "handle_tool_error", _handler)
            except (AttributeError, TypeError):
                pass
        wrapped.append(t)
    return wrapped


def build_rate_limiter() -> InMemoryRateLimiter:
    # Default spacing is ~17s between requests to avoid TPM 429s; override via env
    min_seconds = 17.0
    try:
        env_val = os.getenv("OPENAI_MIN_SECONDS_BETWEEN_REQUESTS")
        if env_val:
            min_seconds = max(0.1, float(env_val))
    except ValueError:
        pass
    rps = 1.0 / min_seconds
    return InMemoryRateLimiter(
        requests_per_second=rps,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
