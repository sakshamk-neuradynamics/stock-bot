from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, UTC
from pathlib import Path
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    get_args,
    get_origin,
)

from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import ValidationError

from . import config
from .tools import create_assemble_report_tool, create_fmp_tools


def _log(title: str, message: str) -> None:
    print(f"[{title}] {message}")


def run_assemble_report_smoke_test() -> bool:
    _log("ASSEMBLE", "Starting assemble_report smoke test")
    workspace_dir = config.WORKSPACE_DIR
    workspace_dir.mkdir(parents=True, exist_ok=True)
    tool = create_assemble_report_tool(workspace_dir)
    demo_root = workspace_dir / "smoke_tests"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    section_dir = demo_root / f"run_{timestamp}"
    section_dir.mkdir(parents=True, exist_ok=True)
    intro = section_dir / "intro.md"
    outlook = section_dir / "outlook.md"
    intro.write_text("Initial findings go here.\n", encoding="utf-8")
    outlook.write_text("Forward-looking notes.\n", encoding="utf-8")
    try:
        message = tool.func(
            section_headings=["Introduction", "Outlook"],
            section_paths=[str(intro), str(outlook.relative_to(workspace_dir))],
        )
        _log("ASSEMBLE", f"Success: {message}")
        return True
    except Exception as exc:
        _log("ASSEMBLE", f"FAILED: {exc}")
        return False


def run_fmp_smoke_tests(symbol: str, period: str, limit: int) -> bool:
    if not config.FMP_API_KEY:
        _log("FMP", "Skipping FMP checks (missing FMP_API_KEY)")
        return True
    _log("FMP", f"Starting FMP smoke tests for {symbol}")
    try:
        tools = create_fmp_tools(
            api_key=config.FMP_API_KEY,
            base_url=config.FMP_BASE_URL,
            timeout=config.FMP_HTTP_TIMEOUT,
        )
    except RuntimeError as exc:
        _log("FMP", f"FAILED to initialize: {exc}")
        return False

    current_year = datetime.now().year
    sample_payloads = {
        "fmp_segments": {
            "symbol": symbol,
            "period": period,
            "structure": "hierarchical",
            "limit": limit,
            "dimension": "product",
        },
        "fmp_footnote_tables": {
            "symbol": symbol,
            "filing_type": "10-K",
            "period": period,
            "year": current_year,
            "limit": 1,
            "include_raw": False,
        },
        "fmp_clean_fundamentals": {
            "symbol": symbol,
            "period": period,
            "limit": limit,
        },
        "fmp_ratios_metrics": {
            "symbol": symbol,
            "period": period,
            "limit": limit,
            "include_growth": False,
        },
        "fmp_balance_sheet": {
            "symbol": symbol,
            "period": "annual",
            "limit": min(5, limit),
        },
        "fmp_dividend_adjusted_prices": {
            "symbol": symbol,
            "from_date": None,
            "to_date": None,
            "limit": max(30, min(200, limit)),
        },
        "fmp_dividends": {
            "symbol": symbol,
            "limit": max(5, min(100, limit)),
        },
        "fmp_structured_sec": {
            "symbol": symbol,
            "cik": None,
            "form_type": "10-K",
            "page": 0,
            "limit": max(1, limit),
            "include_raw": False,
        },
    }

    success = True
    for tool in tools:
        payload = sample_payloads.get(tool.name)
        if not payload:
            _log("FMP", f"Skipping {tool.name} (no sample payload configured)")
            continue
        try:
            result = tool.func(**payload)
            preview = str(result)[:200].replace("\n", " ")
            _log("FMP", f"{tool.name} OK → {preview}")
        except ToolException as exc:
            success = False
            _log("FMP", f"{tool.name} tool error: {exc}")
        except Exception as exc:  # pragma: no cover - network/path errors
            success = False
            _log("FMP", f"{tool.name} FAILED: {exc}")
    return success


async def run_mcp_smoke_tests(
    symbol: str,
    query: str,
    extract_url: str,
    max_tools: int,
) -> bool:
    _log("MCP", "Starting MCP smoke tests")
    client = MultiServerMCPClient(config.MCP_SERVERS)
    try:
        tools = await client.get_tools()
    except Exception as exc:  # pragma: no cover - network errors
        _log("MCP", f"FAILED to fetch tools: {exc}")
        return False

    if not tools:
        _log("MCP", "No MCP tools discovered")
        return False

    attempts = 0
    successes = 0
    for tool in tools:
        if attempts >= max_tools:
            break
        payload = _build_payload_from_schema(tool, symbol, query, extract_url)
        if payload is None:
            _log("MCP", f"Skipping {tool.name} (unable to auto-build payload)")
            continue
        attempts += 1
        try:
            result = await _invoke_tool(tool, payload)
            preview = str(result)[:200].replace("\n", " ")
            _log("MCP", f"{tool.name} OK → {preview}")
            successes += 1
        except ToolException as exc:
            _log("MCP", f"{tool.name} tool error: {exc}")
        except Exception as exc:  # pragma: no cover - network/errors
            _log("MCP", f"{tool.name} FAILED: {exc}")

    close_method = getattr(client, "close", None)
    if callable(close_method):
        maybe = close_method()
        if asyncio.iscoroutine(maybe):
            await maybe
    if successes == 0:
        _log("MCP", "No MCP tools executed successfully")
        return False
    return True


def _build_payload_from_schema(
    tool: StructuredTool,
    symbol: str,
    query: str,
    extract_url: str,
) -> Optional[Dict[str, Any]]:
    schema_cls = getattr(tool, "args_schema", None)
    if schema_cls is None:
        return {}
    if isinstance(schema_cls, dict):
        return _build_payload_from_json_schema(
            schema_cls, symbol=symbol, query=query, extract_url=extract_url
        )

    baseline: Dict[str, Any] = {}
    try:
        instance = schema_cls()
    except TypeError:
        # Non-callable schema; skip this tool.
        return None
    except ValidationError:
        instance = None

    if instance is not None:
        if hasattr(instance, "model_dump"):
            baseline = instance.model_dump(exclude_unset=True)
        elif hasattr(instance, "dict"):
            baseline = instance.dict(exclude_unset=True)

    payload = dict(baseline)
    for name, field in _iter_model_fields(schema_cls):
        if name in payload:
            continue
        guess = _guess_field_value(name, field, symbol, query, extract_url)
        if guess is None:
            if _field_is_required(field):
                return None
            continue
        payload[name] = guess
    return payload


def _iter_model_fields(model: Any) -> Iterable[Tuple[str, Any]]:
    if hasattr(model, "model_fields"):
        return model.model_fields.items()
    if hasattr(model, "__fields__"):
        return model.__fields__.items()
    return []


def _field_is_required(field: Any) -> bool:
    required = getattr(field, "is_required", None)
    if callable(required):
        return bool(required())
    return bool(getattr(field, "required", False))


def _field_annotation(field: Any) -> Any:
    if hasattr(field, "annotation"):
        return field.annotation
    if hasattr(field, "_annotation"):
        return field._annotation  # type: ignore[attr-defined]
    return getattr(field, "outer_type_", None)


def _guess_field_value(
    name: str,
    field: Any,
    symbol: str,
    query: str,
    extract_url: str,
) -> Any:
    guess = getattr(field, "default", None)
    if guess not in (None, Ellipsis):
        return guess
    lname = name.lower()
    if "symbol" in lname or "ticker" in lname:
        return symbol
    if "query" in lname:
        return query
    if "url" in lname:
        return extract_url
    if "limit" in lname:
        return 1
    if "page_size" in lname:
        return 5
    if lname == "page":
        return 0
    if "period" in lname:
        return "annual"
    if "structure" in lname:
        return "hierarchical"
    if "filing" in lname and "type" in lname:
        return "10-K"

    annotation = _field_annotation(field)
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if origin in (list, List, Sequence):
            return []
        if origin is Literal and args:
            return args[0]
        if origin in (tuple, Tuple) and args:
            return tuple()

    if annotation in (int,):
        return 1
    if annotation in (float,):
        return 1.0
    if annotation is bool:
        return False
    if annotation is str:
        return query
    return None


@dataclass
class _JsonField:
    schema: Dict[str, Any]
    required: bool

    @property
    def default(self) -> Any:
        if "default" in self.schema:
            return self.schema["default"]
        enums = self.schema.get("enum")
        if isinstance(enums, list) and enums:
            return enums[0]
        return None

    def is_required(self) -> bool:
        return self.required

    @property
    def _annotation(self) -> Any:
        return _json_schema_annotation(self.schema)


def _json_schema_annotation(schema: Dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return Any


def _build_payload_from_json_schema(
    schema: Dict[str, Any],
    symbol: str,
    query: str,
    extract_url: str,
) -> Optional[Dict[str, Any]]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    required_fields = set(schema.get("required") or [])
    payload: Dict[str, Any] = {}
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            continue
        field = _JsonField(schema=spec, required=name in required_fields)
        guess = _guess_field_value(name, field, symbol, query, extract_url)
        if guess is None and field.is_required():
            return None
        if guess is not None:
            payload[name] = guess
    return payload


async def _invoke_tool(tool: StructuredTool, payload: Dict[str, Any]) -> Any:
    coro = getattr(tool, "coroutine", None)
    if callable(coro):
        return await coro(**payload)
    func = getattr(tool, "func", None)
    if callable(func):
        return await asyncio.to_thread(func, **payload)
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(payload)
    if hasattr(tool, "invoke"):
        return await asyncio.to_thread(tool.invoke, payload)
    raise RuntimeError(f"Tool {tool.name} is missing callable entrypoints.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run live smoke tests against Stock KB tools and configured MCP servers. "
            "These tests make real network calls when API keys are provided."
        )
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol for live requests.")
    parser.add_argument(
        "--period",
        default="annual",
        choices=["annual", "quarter"],
        help="Historical cadence for FMP calls.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=1,
        help="Number of historical periods to request from FMP tools.",
    )
    parser.add_argument(
        "--query",
        default="latest Apple annual report summary",
        help="Query text to use for MCP search-like tools.",
    )
    parser.add_argument(
        "--extract-url",
        default="https://www.apple.com/newsroom/",
        help="URL used when MCP extract tools require one.",
    )
    parser.add_argument(
        "--max-mcp-tools",
        type=int,
        default=3,
        help="Maximum number of MCP tools to execute per run.",
    )
    parser.add_argument(
        "--skip-assemble",
        action="store_true",
        help="Skip the assemble_report smoke test.",
    )
    parser.add_argument(
        "--skip-fmp",
        action="store_true",
        help="Skip tests that call the Financial Modeling Prep API.",
    )
    parser.add_argument(
        "--skip-mcp",
        action="store_true",
        help="Skip MCP tool checks.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    overall_success = True

    if not args.skip_assemble:
        overall_success &= run_assemble_report_smoke_test()

    if not args.skip_fmp:
        overall_success &= run_fmp_smoke_tests(
            symbol=args.symbol,
            period=args.period,
            limit=max(1, args.history_limit),
        )

    if not args.skip_mcp:
        overall_success &= asyncio.run(
            run_mcp_smoke_tests(
                symbol=args.symbol,
                query=args.query,
                extract_url=args.extract_url,
                max_tools=max(1, args.max_mcp_tools),
            )
        )

    if not overall_success:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    main()

