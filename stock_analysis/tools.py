"""Custom LangChain tools for Stock KB agents."""

# pylint: disable=no-self-argument  # Pydantic validators intentionally omit "self".

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field


DEFAULT_FMP_BASE_URL = "https://financialmodelingprep.com/api"


def create_assemble_report_tool(workspace_dir: Path) -> StructuredTool:
    """Create a LangChain tool that assembles report/report.md from per-section files."""
    workspace_dir = workspace_dir.resolve()

    class AssembleReportArgs(BaseModel):
        section_headings: List[str] = Field(
            ..., description="Ordered markdown headings for the sections."
        )
        section_paths: List[str] = Field(
            ...,
            description=(
                "Ordered absolute or workspace-relative paths to files containing each "
                "section's content (e.g., report/cashflow.md)."
            ),
        )

    def _resolve_path(raw: str) -> Path:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = workspace_dir / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(workspace_dir)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Path must reside inside the workspace: {raw}") from exc
        return candidate

    def _read_content(path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "_Section file not found_"
        except UnicodeDecodeError:
            return "_Section file not readable (encoding error)_"
        stripped = text.strip()
        return stripped if stripped else "_Section file is empty_"

    def _assemble(section_headings: List[str], section_paths: List[str]) -> str:
        if len(section_headings) != len(section_paths):
            raise ValueError("section_headings and section_paths must be the same length.")

        resolved_paths = [_resolve_path(raw) for raw in section_paths]
        assembled_blocks: List[str] = []
        for idx, (heading, path) in enumerate(zip(section_headings, resolved_paths), start=1):
            raw_title = heading.strip() or f"Section {idx}"
            title = raw_title if raw_title.startswith("#") else f"# {raw_title}"
            content = _read_content(path)
            block = f"{title}\n\n{content.strip()}"
            assembled_blocks.append(block.strip())

        report_text = "\n\n---\n\n".join(assembled_blocks).strip() + "\n"
        report_dir = workspace_dir / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")

        return (
            f"report/report.md assembled with {len(assembled_blocks)} section(s): "
            + ", ".join(section_headings)
        )

    return StructuredTool.from_function(
        name="assemble_report",
        description=(
            "Assemble report/report.md by pairing ordered section headings with "
            "filesystem paths to each section's markdown content. The tool reads each "
            "file, inserts the heading, and rewrites report/report.md deterministically."
        ),
        func=_assemble,
        args_schema=AssembleReportArgs,
    )


class FinancialModelingPrepClient:
    """Thin HTTP client for the Financial Modeling Prep REST API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_FMP_BASE_URL,
        timeout: float = 30.0,
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("Financial Modeling Prep API key is required.")
        self.api_key = api_key.strip()
        self.base_url = (base_url or DEFAULT_FMP_BASE_URL).rstrip("/")
        self.timeout = timeout

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if not path or not path.strip():
            raise ValueError("A valid API path must be provided.")
        if path.startswith("http://") or path.startswith("https://"):
            url = path
        else:
            url = f"{self.base_url}/{path.lstrip('/')}"
        query = {k: v for k, v in (params or {}).items() if v not in (None, "", [])}
        query["apikey"] = self.api_key
        encoded = urlencode(query, doseq=True)
        target = f"{url}?{encoded}" if encoded else url
        request = Request(
            target,
            headers={
                "Accept": "application/json",
                "User-Agent": "Stock-KB-FMP-Client/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                charset = response.headers.get_content_charset() or "utf-8"
        except HTTPError as exc:
            raise RuntimeError(f"FMP HTTP error ({exc.code}): {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"FMP network error: {exc.reason}") from exc
        try:
            payload = json.loads(raw.decode(charset or "utf-8"))
        except json.JSONDecodeError as exc:
            snippet = raw[:200]
            raise RuntimeError(f"FMP response was not valid JSON: {snippet!r}") from exc
        if isinstance(payload, dict):
            for key in ("error", "Error Message", "message"):
                message = payload.get(key)
                if isinstance(message, str) and message.strip():
                    raise RuntimeError(f"FMP error: {message.strip()}")
        return payload


def create_fmp_tools(
    api_key: Optional[str],
    base_url: str = DEFAULT_FMP_BASE_URL,
    timeout: float = 30.0,
) -> List[StructuredTool]:
    """Instantiate custom FMP tools that cover segments, footnotes, fundamentals, ratios, and SEC data."""

    key = (api_key or "").strip()
    if not key:
        raise RuntimeError(
            "FMP_API_KEY is not configured. Set it before building Financial Modeling Prep tools."
        )

    client = FinancialModelingPrepClient(api_key=key, base_url=base_url, timeout=timeout)
    builders = [
        _build_fmp_segments_tool,
        _build_fmp_footnotes_tool,
        _build_fmp_fundamentals_tool,
        _build_fmp_ratios_tool,
        _build_fmp_sec_tool,
    ]
    return [builder(client) for builder in builders]


def _build_fmp_segments_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class SegmentArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., AAPL.")
        period: Literal["annual", "quarter"] = Field(
            "annual", description="Reporting cadence for the filings to inspect."
        )
        structure: Literal["hierarchical", "flat"] = Field(
            "hierarchical",
            description=(
                "Return nested hierarchy (default) or a flattened table of segment rows."
            ),
        )
        limit: int = Field(
            8,
            ge=1,
            le=40,
            description="Maximum number of filings/periods to return.",
        )

    def _normalize_symbol_input(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty.")
        return cleaned

    def _run(symbol: str, period: str, structure: str, limit: int) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        try:
            payload = client.get(
                "v4/segments",
                {
                    "symbol": normalized_symbol,
                    "period": "quarter" if period == "quarter" else "annual",
                    "structure": "flat" if structure == "flat" else "hierarchy",
                    "limit": limit,
                },
            )
            entries = _ensure_record_list(payload)
            segments: List[Dict[str, Any]] = []
            for entry in entries[:limit]:
                segment_payload = (
                    entry.get("segments")
                    or entry.get("data")
                    or entry.get("details")
                    or entry.get("segmentData")
                )
                segments.append(
                    _strip_nones(
                        {
                            "symbol": entry.get("symbol") or normalized_symbol,
                            "cik": entry.get("cik"),
                            "calendarYear": entry.get("calendarYear"),
                            "period": entry.get("period"),
                            "filed_date": entry.get("filedDate") or entry.get("fillingDate"),
                            "accepted_date": entry.get("acceptedDate"),
                            "segment_data": segment_payload,
                        }
                    )
                )
        except Exception as exc:  # pragma: no cover - HTTP/IO errors
            raise ToolException(str(exc)) from exc

        return {
            "symbol": normalized_symbol,
            "period": period,
            "structure": structure,
            "count": len(segments),
            "records": segments,
            "source": "Financial Modeling Prep v4/segments",
        }

    return StructuredTool.from_function(
        name="fmp_segments",
        description=(
            "Fetch detailed segment-level revenue/margin breakdowns directly from "
            "Financial Modeling Prep's v4/segments endpoint."
        ),
        func=_run,
        args_schema=SegmentArgs,
    )


def _build_fmp_footnotes_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class FootnoteArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., MSFT.")
        filing_type: Literal["10-K", "10-Q"] = Field(
            "10-K", description="Filing type to target."
        )
        period: Literal["annual", "quarter"] = Field(
            "annual", description="Whether to scan annual or quarterly filings."
        )
        year: Optional[int] = Field(
            None,
            ge=1994,
            le=2100,
            description="Optional fiscal/calendar year filter.",
        )
        limit: int = Field(
            2,
            ge=1,
            le=8,
            description="Maximum number of filings to return.",
        )
        include_raw: bool = Field(
            False,
            description="Also return the raw footnotes JSON payload for each filing.",
        )

    def _normalize_symbol_input(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty.")
        return cleaned

    def _run(
        symbol: str,
        filing_type: str,
        period: str,
        year: Optional[int],
        limit: int,
        include_raw: bool,
    ) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        try:
            payload = client.get(
                "v4/financial-reports-json",
                {
                    "symbol": normalized_symbol,
                    "reportType": filing_type,
                    "period": "quarter" if period == "quarter" else "annual",
                    "year": year,
                    "limit": limit,
                },
            )
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        entries = _ensure_record_list(payload)
        filings: List[Dict[str, Any]] = []
        for entry in entries:
            if len(filings) >= limit:
                break
            entry_type = str(entry.get("reportType") or "").upper()
            if entry_type and entry_type != filing_type:
                continue
            tables = _parse_footnote_tables(entry)
            filing_record = _strip_nones(
                {
                    "symbol": entry.get("symbol") or normalized_symbol,
                    "cik": entry.get("cik"),
                    "report_type": entry.get("reportType"),
                    "form": entry.get("form"),
                    "calendarYear": entry.get("calendarYear") or entry.get("year"),
                    "period": entry.get("period"),
                    "filed_date": entry.get("filedDate") or entry.get("fillingDate"),
                    "accepted_date": entry.get("acceptedDate"),
                    "footnote_tables": tables,
                    "source": entry.get("finalLink") or entry.get("link"),
                }
            )
            if include_raw:
                filing_record["raw_footnotes"] = entry.get("footnotes")
            filings.append(filing_record)

        return {
            "symbol": normalized_symbol,
            "filing_type": filing_type,
            "period": period,
            "count": len(filings),
            "filings": filings,
            "source": "Financial Modeling Prep v4/financial-reports-json",
        }

    return StructuredTool.from_function(
        name="fmp_footnote_tables",
        description=(
            "Extract normalized 10-K/10-Q footnote tables from Financial Modeling Prep's "
            "financial-reports-json dataset."
        ),
        func=_run,
        args_schema=FootnoteArgs,
    )


def _build_fmp_fundamentals_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class FundamentalsArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., NVDA.")
        period: Literal["annual", "quarter"] = Field(
            "annual", description="Reporting cadence for the normalized statements."
        )
        limit: int = Field(
            5,
            ge=1,
            le=20,
            description="Number of historical periods to merge (Income, Balance, Cash).",
        )

    def _normalize_symbol_input(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty.")
        return cleaned

    def _run(symbol: str, period: str, limit: int) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        cadence = "quarter" if period == "quarter" else "annual"
        rows: Dict[str, Dict[str, Any]] = {}
        try:
            income = _ensure_record_list(
                client.get(
                    f"v3/income-statement/{normalized_symbol}",
                    {"period": cadence, "limit": limit},
                )
            )
            balance = _ensure_record_list(
                client.get(
                    f"v3/balance-sheet-statement/{normalized_symbol}",
                    {"period": cadence, "limit": limit},
                )
            )
            cash = _ensure_record_list(
                client.get(
                    f"v3/cash-flow-statement/{normalized_symbol}",
                    {"period": cadence, "limit": limit},
                )
            )
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        _merge_section_data(rows, income, "income_statement", INCOME_FIELDS)
        _merge_section_data(rows, balance, "balance_sheet", BALANCE_FIELDS)
        _merge_section_data(rows, cash, "cash_flow", CASH_FIELDS)
        records = _sorted_period_records(rows, limit)

        return {
            "symbol": normalized_symbol,
            "period": period,
            "records": records,
            "source": "Financial Modeling Prep v3 statements",
        }

    return StructuredTool.from_function(
        name="fmp_clean_fundamentals",
        description=(
            "Return clean, merged historical fundamentals by stitching the Income "
            "Statement, Balance Sheet, and Cash Flow statement for each filing period."
        ),
        func=_run,
        args_schema=FundamentalsArgs,
    )


def _build_fmp_ratios_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class RatioArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., GOOG.")
        period: Literal["annual", "quarter"] = Field(
            "annual", description="Whether to sample annual or quarterly filings."
        )
        limit: int = Field(
            5,
            ge=1,
            le=20,
            description="Max number of periods to return.",
        )
        include_growth: bool = Field(
            True,
            description="Include FMP financial-growth metrics derived from filings.",
        )

    def _normalize_symbol_input(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty.")
        return cleaned

    def _run(symbol: str, period: str, limit: int, include_growth: bool) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc
        cadence = "quarter" if period == "quarter" else "annual"
        params = {"period": cadence, "limit": limit}
        try:
            ratios = _ensure_record_list(client.get(f"v3/ratios/{normalized_symbol}", params))
            metrics = _ensure_record_list(client.get(f"v3/key-metrics/{normalized_symbol}", params))
            growth = (
                _ensure_record_list(client.get(f"v3/financial-growth/{normalized_symbol}", params))
                if include_growth
                else []
            )
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        rows: Dict[str, Dict[str, Any]] = {}
        _merge_section_data(rows, ratios, "ratios", RATIO_FIELDS)
        _merge_section_data(rows, metrics, "key_metrics", KEY_METRIC_FIELDS)
        if include_growth:
            _merge_section_data(rows, growth, "growth", GROWTH_FIELDS)
        records = _sorted_period_records(rows, limit)

        return {
            "symbol": normalized_symbol,
            "period": period,
            "records": records,
            "source": "Financial Modeling Prep v3 ratios/key-metrics",
        }

    return StructuredTool.from_function(
        name="fmp_ratios_metrics",
        description=(
            "Compute filing-derived ratios, key metrics, and optional YoY growth using "
            "Financial Modeling Prep's ratios/key-metrics APIs."
        ),
        func=_run,
        args_schema=RatioArgs,
    )


def _build_fmp_sec_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class SecArgs(BaseModel):
        symbol: Optional[str] = Field(
            None,
            description="Ticker symbol (provide symbol or CIK).",
        )
        cik: Optional[str] = Field(
            None,
            description="SEC CIK with or without leading zeros.",
        )
        form_type: Optional[str] = Field(
            None,
            description="Filter by SEC form type, e.g., 10-K, 10-Q, 8-K.",
        )
        page: int = Field(0, ge=0, description="Pagination offset.")
        page_size: int = Field(
            40,
            ge=1,
            le=200,
            description="Number of filings per page.",
        )
        include_raw: bool = Field(
            False,
            description="Include the raw SEC filing payload returned by FMP.",
        )

    def _normalize_symbol_input(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        return cleaned or None

    def _normalize_cik_input(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("CIK must be a string.")
        cleaned = value.strip().lstrip("0")
        return cleaned or None

    def _run(
        symbol: Optional[str],
        cik: Optional[str],
        form_type: Optional[str],
        page: int,
        page_size: int,
        include_raw: bool,
    ) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
            normalized_cik = _normalize_cik_input(cik)
            if not normalized_symbol and not normalized_cik:
                raise ValueError("Provide at least a symbol or a CIK.")
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        try:
            payload = client.get(
                "v3/sec_filings",
                {
                    "symbol": normalized_symbol,
                    "cik": normalized_cik,
                    "type": form_type,
                    "page": page,
                    "pageSize": page_size,
                },
            )
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        entries = _ensure_record_list(payload)
        filings: List[Dict[str, Any]] = []
        for entry in entries[:page_size]:
            record = _strip_nones(
                {
                    "symbol": entry.get("symbol") or normalized_symbol,
                    "cik": entry.get("cik") or normalized_cik,
                    "form_type": entry.get("type") or entry.get("form"),
                    "filed_date": entry.get("fillingDate") or entry.get("filedDate"),
                    "accepted_date": entry.get("acceptedDate"),
                    "report_period": entry.get("period") or entry.get("periodOfReport"),
                    "report_url": entry.get("finalLink") or entry.get("link"),
                    "source": entry.get("source"),
                }
            )
            if include_raw:
                record["raw"] = entry
            filings.append(record)

        return {
            "symbol": normalized_symbol,
            "cik": normalized_cik,
            "form_type": form_type,
            "page": page,
            "count": len(filings),
            "filings": filings,
            "source": "Financial Modeling Prep v3/sec_filings",
        }

    return StructuredTool.from_function(
        name="fmp_structured_sec",
        description=(
            "Access a stable, structured feed of SEC filings (metadata + links) via "
            "Financial Modeling Prep's sec_filings endpoint."
        ),
        func=_run,
        args_schema=SecArgs,
    )


def _ensure_record_list(payload: Any) -> List[Dict[str, Any]]:
    """Coerce heterogeneous API responses into a list of dicts."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "financials", "filings", "reports"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def _strip_nones(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v is not None}


INCOME_FIELDS: Tuple[str, ...] = (
    "revenue",
    "grossProfit",
    "operatingIncome",
    "netIncome",
    "ebit",
    "ebitda",
    "eps",
    "epsdiluted",
)

BALANCE_FIELDS: Tuple[str, ...] = (
    "totalAssets",
    "totalLiabilities",
    "totalDebt",
    "cashAndCashEquivalents",
    "shortTermInvestments",
    "longTermDebt",
    "netDebt",
    "shareholderEquity",
    "retainedEarnings",
)

CASH_FIELDS: Tuple[str, ...] = (
    "netCashProvidedByOperatingActivities",
    "netCashUsedForInvestingActivites",
    "capitalExpenditure",
    "freeCashFlow",
    "dividendsPaid",
)

RATIO_FIELDS: Tuple[str, ...] = (
    "priceEarningsRatio",
    "priceToBookRatio",
    "priceToSalesRatio",
    "priceCashFlowRatio",
    "debtEquityRatio",
    "returnOnEquity",
    "returnOnAssets",
    "returnOnCapitalEmployed",
    "grossProfitMargin",
    "operatingProfitMargin",
    "netProfitMargin",
    "currentRatio",
)

KEY_METRIC_FIELDS: Tuple[str, ...] = (
    "enterpriseValue",
    "marketCap",
    "peRatio",
    "pegRatio",
    "payoutRatio",
    "evToSales",
    "evToOperatingCashFlow",
    "evToEbitda",
    "priceToFreeCashFlowsRatio",
    "priceToOperatingCashFlowsRatio",
    "priceToBookRatio",
    "priceToSalesRatio",
)

GROWTH_FIELDS: Tuple[str, ...] = (
    "revenueGrowth",
    "grossProfitGrowth",
    "ebitgrowth",
    "operatingIncomeGrowth",
    "netIncomeGrowth",
    "epsgrowth",
    "freeCashFlowGrowth",
    "totalAssetsGrowth",
    "bookValueperShareGrowth",
)


def _merge_section_data(
    store: Dict[str, Dict[str, Any]],
    rows: Iterable[Dict[str, Any]],
    section: str,
    fields: Sequence[str],
) -> None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = _period_key(row)
        if not key:
            continue
        bucket = store.setdefault(
            key,
            {
                "date": row.get("date") or row.get("fiscalDateEnding"),
                "calendarYear": row.get("calendarYear"),
                "period": row.get("period"),
                "reportedCurrency": row.get("reportedCurrency"),
            },
        )
        trimmed = _trim_fields(row, fields)
        if trimmed:
            bucket[section] = trimmed


def _period_key(row: Dict[str, Any]) -> Optional[str]:
    for key in ("date", "fiscalDateEnding", "periodEndDate", "filing_date"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    calendar_year = row.get("calendarYear")
    period = row.get("period")
    if calendar_year and period:
        return f"{calendar_year}-{period}"
    return None


def _trim_fields(record: Optional[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    trimmed: Dict[str, Any] = {}
    for field_name in fields:
        value = record.get(field_name)
        if value is not None:
            trimmed[field_name] = value
    return trimmed


def _sorted_period_records(store: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    records = list(store.values())
    records.sort(key=lambda rec: (rec.get("date") or "", rec.get("calendarYear") or ""), reverse=True)
    return records[:limit]


def _parse_footnote_tables(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_tables = (
        entry.get("footnotes")
        or entry.get("footnotesTable")
        or entry.get("footnotesTables")
        or entry.get("notes")
    )
    tables: List[Dict[str, Any]] = []
    items: Iterable[Tuple[str, Any]]
    if isinstance(raw_tables, dict):
        items = raw_tables.items()
    elif isinstance(raw_tables, list):
        items = [(str(idx), value) for idx, value in enumerate(raw_tables, start=1)]
    else:
        items = []

    for name, value in items:
        if isinstance(value, dict):
            rows = value.get("rows") or value.get("data") or value.get("table") or value.get("values")
            label = value.get("title") or value.get("label") or name
        else:
            rows = value
            label = name
        tables.append(
            _strip_nones(
                {
                    "label": label,
                    "rows": rows,
                }
            )
        )
    return tables