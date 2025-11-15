from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

from langchain_core.rate_limiters import InMemoryRateLimiter




VALUE_PRINCIPLES_TOKEN = "{{VALUE_INVESTING_PRINCIPLES}}"


def prompts_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


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


