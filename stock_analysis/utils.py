from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Dict
import re
import hashlib
import json

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools.base import ToolException
from langchain_core.tools import StructuredTool

VALUE_PRINCIPLES_TOKEN = "{{VALUE_INVESTING_PRINCIPLES}}"

ALPHA_VANTAGE_COMMODITY_TOOL_NAMES = (
    "natural_gas",
    "copper",
    "aluminum",
    "wheat",
    "corn",
    "cotton",
    "sugar",
    "coffee",
)


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


def filter_out_alpha_vantage_commodities(
    tools: Iterable[Any], extra_names: Optional[Iterable[str]] = None
) -> List[Any]:
    """
    Remove Alpha Vantage commodity tools that we don't want exposed to agents.
    """

    excluded = {name.lower() for name in ALPHA_VANTAGE_COMMODITY_TOOL_NAMES}
    if extra_names:
        excluded.update(name.lower() for name in extra_names)
    return filter_out_tools_by_names(tools, names=excluded)


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
        return (
            ("validation error" in s)
            or ("unexpected keyword argument" in s)
            or ("not one of" in s)
        )

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


# ---------------------------
# Large-output materialization
# ---------------------------


def _slugify_url(url: str) -> str:
    """Create a filesystem-friendly slug from a URL."""
    if not isinstance(url, str) or not url:
        return "doc"
    s = re.sub(r"https?://", "", url, flags=re.IGNORECASE)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s[:80] or "doc"


def _chunk_and_save_markdown(
    text: str, root: Path, slug: str, max_chars: int = 6000
) -> List[Path]:
    """Normalize newlines, chunk text, and save as numbered .md files under root."""
    root.mkdir(parents=True, exist_ok=True)
    clean = re.sub(r"\r\n?", "\n", text or "")
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    # Ensure ends with newline for cleaner diffs
    if not clean.endswith("\n"):
        clean = clean + "\n"
    chunks: List[str] = [
        clean[i : i + max_chars] for i in range(0, len(clean), max_chars)
    ] or [clean]
    paths: List[Path] = []
    for idx, chunk in enumerate(chunks, start=1):
        path = root / f"{slug}_chunk_{idx:03d}.md"
        path.write_text(chunk, encoding="utf-8")
        paths.append(path)
    return paths


def materialize_extract_payload(
    url: Optional[str],
    content: str,
    workspace_dir: Path,
    subdir: str = "websearch/extract",
    max_chars: int = 6000,
) -> Dict[str, Any]:
    """Persist a large extract payload into chunked .md files and return small pointers."""
    slug = _slugify_url(url or "doc")
    out_root = workspace_dir / subdir / slug
    paths = _chunk_and_save_markdown(
        content or "", out_root, slug=slug, max_chars=max_chars
    )
    sha = hashlib.sha1((content or "").encode("utf-8")).hexdigest()
    # Build workspace-relative POSIX paths (e.g., "websearch/extract/.../chunk_001.md")
    rel_paths: List[str] = []
    for p in paths:
        try:
            rel = p.relative_to(workspace_dir).as_posix()
        except ValueError:
            # Fallback to absolute, but normalize separators
            rel = p.as_posix()
        rel_paths.append(rel)

    return {
        "url": url,
        "sha1": sha,
        "chunks": len(paths),
        "paths": rel_paths,
    }


def _extract_text_generic(result: Any) -> str:
    """Best-effort extraction of text content from a tool result."""
    if isinstance(result, dict):
        for key in ("content", "text", "raw", "page_content"):
            val = result.get(key)
            if isinstance(val, str):
                return val
    return str(result) if result is not None else ""


def materialize_extract_result(
    kwargs: Dict[str, Any],
    result: Any,
    workspace_dir: Path,
    max_chars: int = 6000,
) -> Dict[str, Any]:
    """
    Materialize extract outputs when tools may accept multiple URLs via 'urls'.
    Handles common shapes:
      1) result is dict with 'results': [ {url, content|text|raw|page_content}, ... ]
      2) result is list of dicts with url/content
      3) result is a single string/dict → use first URL from kwargs['urls'] if present
    Returns a compact pointer structure with per-item saved paths.
    """
    urls_param: List[str] = []
    u = kwargs.get("urls")
    if isinstance(u, str) and u.strip():
        urls_param = [u.strip()]
    elif isinstance(u, (list, tuple)):
        urls_param = [str(x).strip() for x in u if isinstance(x, str) and x.strip()]

    items_out: List[Dict[str, Any]] = []

    def _pick_text(d: Dict[str, Any]) -> str:
        for k in ("content", "text", "raw", "page_content"):
            v = d.get(k)
            if isinstance(v, str):
                return v
        return ""

    # Case 1: dict with 'results'
    if isinstance(result, dict) and isinstance(result.get("results"), list):
        for entry in result["results"]:
            if not isinstance(entry, dict):
                continue
            url = entry.get("url")
            text = _pick_text(entry) or _extract_text_generic(entry)
            mat = materialize_extract_payload(url, text, workspace_dir, max_chars=max_chars)
            items_out.append(mat)
    # Case 2: list of dicts
    elif isinstance(result, list):
        for entry in result:
            if isinstance(entry, dict):
                url = entry.get("url")
                text = _pick_text(entry) or _extract_text_generic(entry)
                mat = materialize_extract_payload(url, text, workspace_dir, max_chars=max_chars)
                items_out.append(mat)
    # Case 3: single blob
    else:
        primary_url = urls_param[0] if urls_param else None
        text = _extract_text_generic(result)
        mat = materialize_extract_payload(primary_url, text, workspace_dir, max_chars=max_chars)
        items_out.append(mat)

    return {
        "urls": urls_param or [itm.get("url") for itm in items_out if itm.get("url")] or [],
        "items": items_out,
        "chunks_total": sum(int(itm.get("chunks", 0)) for itm in items_out),
        "saved": [p for itm in items_out for p in itm.get("paths", [])],
    }


def wrap_tools_with_extract_materializer(tools: Iterable[Any], workspace_dir: Path, max_chars: int = 6000) -> List[Any]:
    """
    Return a new tools list where each tool response is prettified (if possible) and long payloads are
    persisted to disk with small pointer summaries. This avoids mutating Pydantic tool instances
    (which often disallow setattr) while still constraining LLM response size.
    """

    def _looks_like_extract_response(kwargs: Dict[str, Any], result: Any) -> bool:
        if isinstance(result, dict) and isinstance(result.get("results"), list):
            return True
        if isinstance(result, list) and result:
            for entry in result:
                if isinstance(entry, dict) and entry.get("url"):
                    return True
        return bool(kwargs.get("urls"))

    def _prettify_result_text(result: Any) -> str:
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                return str(result)
        if isinstance(result, str):
            stripped = result.strip()
            if stripped[:1] in {"{", "["}:
                try:
                    parsed = json.loads(result)
                except json.JSONDecodeError:
                    return result
                else:
                    try:
                        return json.dumps(parsed, ensure_ascii=False, indent=2)
                    except (TypeError, ValueError):
                        return result
            return result
        return str(result) if result is not None else ""

    def _materialize_generic(tool_name: str, content: str, workspace_dir: Path) -> Dict[str, Any]:
        slug = _slugify_url(tool_name or "tool")
        pointer = materialize_extract_payload(
            url=None,
            content=content,
            workspace_dir=workspace_dir,
            subdir=f"tool_outputs/{slug}",
            max_chars=max_chars,
        )
        return {
            "message": (
                "Tool output exceeded the inline response limit. "
                "Full content saved under `stock_analysis/workspace/{}`.".format(
                    pointer["paths"][0] if pointer.get("paths") else "tool_outputs"
                )
            ),
            "materialized": pointer,
            "preview": (content or "")[: max_chars // 2],
        }

    def _format_tool_error(tool_name: str, exc: Exception) -> Dict[str, Any]:
        text = str(exc).strip()
        lowered = text.lower()
        if "timeout" in lowered:
            summary = "Tool request timed out. Try again or switch sources."
        elif "connection" in lowered or "network" in lowered:
            summary = "Tool request failed due to a connection error."
        else:
            summary = "Tool request failed."
        detail = text or repr(exc)
        return {
            "error": summary,
            "tool": tool_name,
            "detail": detail[:500],
        }

    def _should_wrap_exception(exc: Exception) -> bool:
        lowered = str(exc).lower()
        return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or "timeout" in lowered or "connection" in lowered

    def _process_result(tool_name: str, kwargs: Dict[str, Any], result: Any):
        formatted = _prettify_result_text(result)
        if len(formatted) <= max_chars:
            return formatted
        if _looks_like_extract_response(kwargs, result):
            return materialize_extract_result(kwargs, result, workspace_dir, max_chars=max_chars)
        return _materialize_generic(tool_name, formatted, workspace_dir)

    wrapped: List[Any] = []
    for tool in tools:

        if getattr(tool, "_materialize_wrapped", False):
            wrapped.append(tool)
            continue

        # If we can construct a new StructuredTool with a wrapped function/coroutine, prefer that.
        orig_func = getattr(tool, "func", None)
        orig_coro = getattr(tool, "coroutine", None)
        args_schema = getattr(tool, "args_schema", None)
        name = getattr(tool, "name", "tavily_extract")
        description = getattr(tool, "description", "") or ""

        new_tool = None
        if StructuredTool is not None and (callable(orig_func) or callable(orig_coro)):
            def wrapped_func(*, orig_func=orig_func, tool_name=name, **kwargs):
                try:
                    result = orig_func(**kwargs) if callable(orig_func) else None
                except Exception as exc:  # pragma: no cover - tool runtime varies
                    if _should_wrap_exception(exc):
                        return _format_tool_error(tool_name, exc)
                    raise
                return _process_result(tool_name, kwargs, result)

            async def wrapped_coro(*, orig_coro=orig_coro, orig_func=orig_func, tool_name=name, **kwargs):
                try:
                    if callable(orig_coro):
                        result = await orig_coro(**kwargs)
                    else:
                        result = orig_func(**kwargs) if callable(orig_func) else None
                except Exception as exc:  # pragma: no cover - tool runtime varies
                    if _should_wrap_exception(exc):
                        return _format_tool_error(tool_name, exc)
                    raise
                return _process_result(tool_name, kwargs, result)

            try:
                new_tool = StructuredTool(
                    name=name,
                    description=description,
                    args_schema=args_schema,
                    func=wrapped_func if callable(orig_func) else None,
                    coroutine=wrapped_coro if callable(orig_coro) else None,
                    # Preserve existing error handler if present
                    handle_tool_error=getattr(tool, "handle_tool_error", None),
                )
            except (TypeError, ValueError):
                new_tool = None

        if new_tool is None:
            # Fallback: cannot rebuild; leave tool unchanged
            wrapped.append(tool)
            continue

        try:
            setattr(new_tool, "_materialize_wrapped", True)
        except (AttributeError, TypeError):
            pass

        wrapped.append(new_tool)
    return wrapped


def build_rate_limiter() -> InMemoryRateLimiter:
    # Default spacing is ~17s between requests to avoid TPM 429s; override via env
    min_seconds = 0.5
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
