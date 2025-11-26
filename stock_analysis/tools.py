"""Custom LangChain tools for Stock KB agents."""

# pylint: disable=no-self-argument,broad-except  # Pydantic validators omit "self"; broad exceptions converted to tool errors.

from __future__ import annotations

import json
from pathlib import Path
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import StructuredTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FMP_BASE_URL = "https://financialmodelingprep.com/api"


def _wrap_tool_func(func):
    def _wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            return str(exc)

    return _wrapper


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

    def _normalize_image_paths(markdown: str, source_path: Path) -> str:
        pattern = re.compile(r'(!\[[^\]]*\]\()([^)\s]+)([^)]*\))')

        def _normalize_relative(url: str) -> str:
            trimmed = url.strip()
            if not trimmed or trimmed.startswith(("http://", "https://", "data:", "file:")):
                return url
            normalized = trimmed.replace("\\", "/")
            normalized = normalized.lstrip("./")
            while normalized.startswith("../"):
                normalized = normalized[3:]
            normalized = normalized.lstrip("/")
            lower_norm = normalized.lower()
            if "report/figures" in lower_norm:
                idx = lower_norm.rfind("report/figures")
                relative = normalized[idx:]
                return relative[len("report/") :]
            if lower_norm.startswith("figures/"):
                return normalized
            return url

        changed = False

        def _replace(match: re.Match[str]) -> str:
            nonlocal changed
            prefix, url, suffix = match.groups()
            new_url = _normalize_relative(url)
            if new_url != url:
                changed = True
            return f"{prefix}{new_url}{suffix}"

        new_text = pattern.sub(_replace, markdown)
        if changed:
            source_path.write_text(new_text, encoding="utf-8")
        return new_text

    def _read_content(path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return "_Section file not found_"
        except UnicodeDecodeError:
            return "_Section file not readable (encoding error)_"
        stripped = text.strip()
        if not stripped:
            return "_Section file is empty_"
        return _normalize_image_paths(stripped, path)

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
            "Use after subagents have produced per-section markdown files and you need a "
            "single report. Supply matching lists of headings and file paths (absolute or "
            "workspace-relative). The tool reads each file, prepends the heading, and "
            "writes the stitched document to report/report.md. Example: headings=['Executive "
            "Summary', 'Valuation'], section_paths=['report/summary.md', 'report/valuation.md']."
        ),
        func=_wrap_tool_func(_assemble),
        args_schema=AssembleReportArgs,
    )


def create_bar_chart_tool(workspace_dir: Path) -> StructuredTool:
    """Create a LangChain tool that renders a bar chart image inside the workspace."""
    workspace_dir = workspace_dir.resolve()

    class BarChartArgs(BaseModel):
        title: Optional[str] = Field(None, description="Optional chart title.")
        categories: List[str] = Field(
            ...,
            min_items=1,
            description="Labels for each bar; length must match values.",
        )
        values: List[float] = Field(
            ...,
            min_items=1,
            description="Numerical values for each category.",
        )
        x_label: Optional[str] = Field(None, description="Label for the x-axis.")
        y_label: Optional[str] = Field(None, description="Label for the y-axis.")
        color: Optional[str] = Field(
            None, description="Matplotlib-compatible color for the bars."
        )
        width: float = Field(8.0, gt=0, description="Figure width in inches.")
        height: float = Field(5.0, gt=0, description="Figure height in inches.")
        rotate_labels: bool = Field(
            False, description="Rotate x-axis labels 45 degrees for readability."
        )
        output_filename: Optional[str] = Field(
            None,
            description=(
                "Optional filename (png) relative to the workspace/report/figures directory."
            ),
        )

    def _run(
        title: Optional[str],
        categories: List[str],
        values: List[float],
        x_label: Optional[str],
        y_label: Optional[str],
        color: Optional[str],
        width: float,
        height: float,
        rotate_labels: bool,
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        plotting = _ensure_matplotlib_ready()
        if len(categories) != len(values):
            raise ValueError("categories and values must be the same length.")

        target_path = _resolve_chart_output_path(workspace_dir, output_filename, "bar")
        fig, ax = plotting.subplots(figsize=(width, height))
        ax.bar(categories, values, color=color)

        if title:
            ax.set_title(title)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if rotate_labels or len(categories) > 8:
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight", facecolor="white")
        plotting.close(fig)
        return {
            "chart_type": "bar",
            "path": str(target_path),
            "points": len(categories),
        }

    return StructuredTool.from_function(
        name="chart_bar",
        description=(
            "Render a bar chart from categories and values, saving the figure into "
            "report/figures and returning the absolute path to the PNG."
        ),
        func=_wrap_tool_func(_run),
        args_schema=BarChartArgs,
    )


def create_line_chart_tool(workspace_dir: Path) -> StructuredTool:
    """Create a LangChain tool that renders a line chart image inside the workspace."""
    workspace_dir = workspace_dir.resolve()

    class LineChartArgs(BaseModel):
        title: Optional[str] = Field(None, description="Optional chart title.")
        x_values: List[Any] = Field(
            ...,
            min_items=2,
            description="X-axis values (strings or numbers) for each point.",
        )
        y_values: List[float] = Field(
            ...,
            min_items=2,
            description="Numeric Y-axis values corresponding to x_values.",
        )
        x_label: Optional[str] = Field(None, description="Label for the x-axis.")
        y_label: Optional[str] = Field(None, description="Label for the y-axis.")
        line_color: Optional[str] = Field(
            None, description="Matplotlib color for the line."
        )
        line_style: str = Field(
            "-",
            description="Matplotlib line style string (e.g., '-', '--', ':').",
        )
        show_markers: bool = Field(
            False, description="Plot a marker at each data point."
        )
        marker: Optional[str] = Field(
            None, description="Optional matplotlib marker style, defaults to 'o'."
        )
        grid: bool = Field(True, description="Show a background grid.")
        width: float = Field(8.0, gt=0, description="Figure width in inches.")
        height: float = Field(5.0, gt=0, description="Figure height in inches.")
        output_filename: Optional[str] = Field(
            None,
            description="Optional filename (png) relative to report/figures.",
        )

    def _run(
        title: Optional[str],
        x_values: List[Any],
        y_values: List[float],
        x_label: Optional[str],
        y_label: Optional[str],
        line_color: Optional[str],
        line_style: str,
        show_markers: bool,
        marker: Optional[str],
        grid: bool,
        width: float,
        height: float,
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        plotting = _ensure_matplotlib_ready()
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must be the same length.")
        if len(x_values) < 2:
            raise ValueError("Provide at least two points for a line chart.")

        target_path = _resolve_chart_output_path(workspace_dir, output_filename, "line")
        fig, ax = plotting.subplots(figsize=(width, height))
        marker_style = marker or ("o" if show_markers else None)
        ax.plot(
            x_values,
            y_values,
            color=line_color,
            linestyle=line_style or "-",
            marker=marker_style,
        )

        if title:
            ax.set_title(title)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if grid:
            ax.grid(True, linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight", facecolor="white")
        plotting.close(fig)
        return {
            "chart_type": "line",
            "path": str(target_path),
            "points": len(x_values),
        }

    return StructuredTool.from_function(
        name="chart_line",
        description=(
            "Render a line chart from paired x/y inputs, save it to report/figures, and "
            "return the absolute file path."
        ),
        func=_wrap_tool_func(_run),
        args_schema=LineChartArgs,
    )


def create_pie_chart_tool(workspace_dir: Path) -> StructuredTool:
    """Create a LangChain tool that renders a pie chart image inside the workspace."""
    workspace_dir = workspace_dir.resolve()

    class PieChartArgs(BaseModel):
        title: Optional[str] = Field(None, description="Optional chart title.")
        labels: List[str] = Field(
            ...,
            min_items=1,
            description="Labels for each slice; length must match values.",
        )
        values: List[float] = Field(
            ...,
            min_items=1,
            description="Numeric values for each slice.",
        )
        colors: Optional[List[str]] = Field(
            None,
            description="Optional list of colors, one per slice.",
        )
        explode: Optional[List[float]] = Field(
            None,
            description="Optional explode offsets (0-1) for each slice.",
        )
        start_angle: float = Field(
            90.0, description="Starting angle in degrees (default 90)."
        )
        show_percentages: bool = Field(
            True, description="Display percentage labels on each slice."
        )
        autopct_format: str = Field(
            "%1.1f%%", description="Format string used when show_percentages is true."
        )
        show_legend: bool = Field(
            False, description="Add a legend to the right of the pie chart."
        )
        output_filename: Optional[str] = Field(
            None,
            description="Optional filename (png) relative to report/figures.",
        )

    def _run(
        title: Optional[str],
        labels: List[str],
        values: List[float],
        colors: Optional[List[str]],
        explode: Optional[List[float]],
        start_angle: float,
        show_percentages: bool,
        autopct_format: str,
        show_legend: bool,
        output_filename: Optional[str],
    ) -> Dict[str, Any]:
        plotting = _ensure_matplotlib_ready()
        if len(labels) != len(values):
            raise ValueError("labels and values must be the same length.")
        if colors and len(colors) != len(values):
            raise ValueError("colors must match the number of values.")
        if explode and len(explode) != len(values):
            raise ValueError("explode must match the number of values.")
        if sum(values) <= 0:
            raise ValueError("values must sum to a positive number.")

        target_path = _resolve_chart_output_path(workspace_dir, output_filename, "pie")
        fig, ax = plotting.subplots(figsize=(6, 6))
        pie_kwargs: Dict[str, Any] = {
            "labels": labels,
            "startangle": start_angle,
            "autopct": autopct_format if show_percentages else None,
        }
        if colors:
            pie_kwargs["colors"] = colors
        if explode:
            pie_kwargs["explode"] = explode

        wedges, _texts, autotexts = ax.pie(values, **pie_kwargs)
        if not show_percentages:
            autotexts = []
        for autotext in autotexts or []:
            autotext.set_color("white")
            autotext.set_fontweight("bold")

        ax.axis("equal")
        if title:
            ax.set_title(title)
        if show_legend:
            ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        fig.savefig(target_path, dpi=300, bbox_inches="tight", facecolor="white")
        plotting.close(fig)
        return {
            "chart_type": "pie",
            "path": str(target_path),
            "slices": len(labels),
        }

    return StructuredTool.from_function(
        name="chart_pie",
        description=(
            "Render a pie chart from labels and values, store it under report/figures, "
            "and return the absolute path to the saved PNG."
        ),
        func=_wrap_tool_func(_run),
        args_schema=PieChartArgs,
    )


def create_chart_tools(workspace_dir: Path) -> List[StructuredTool]:
    """Convenience helper that returns bar, line, and pie chart tools."""
    return [
        create_bar_chart_tool(workspace_dir),
        create_line_chart_tool(workspace_dir),
        create_pie_chart_tool(workspace_dir),
    ]


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
        _build_fmp_balance_sheet_tool,
        _build_fmp_dividend_adjusted_prices_tool,
        _build_fmp_dividends_tool,
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
            "flat",
            description=(
                "Return nested hierarchy (default) or a flattened table of segment rows."
            ),
        )
        dimension: Literal["product", "geographic"] = Field(
            "product",
            description="Choose product-based or geographic-based revenue segmentation.",
        )

    def _normalize_symbol_input(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty.")
        return cleaned

    def _run(
        symbol: str, period: str, structure: str, limit: int, dimension: str
    ) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        try:
            capped_limit = max(1, limit)
            if dimension == "geographic":
                endpoint = "revenue-geographic-segmentation"
            else:
                endpoint = "revenue-product-segmentation"
            cadence = "quarter" if period == "quarter" else "annual"
            payload = client.get(
                endpoint,
                {
                    "symbol": normalized_symbol,
                    "period": cadence,
                    "structure": "flat",
                },
            )
            entries = _ensure_record_list(payload)
            segments: List[Dict[str, Any]] = []
            for entry in entries:
                if len(segments) >= capped_limit:
                    break
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
            "dimension": dimension,
            "count": len(segments),
            "records": segments,
            "source": f"Financial Modeling Prep stable/{endpoint}",
        }

    return StructuredTool.from_function(
        name="fmp_segments",
        description=(
            "Fetch product- or geography-level revenue splits from Financial Modeling "
            "Prep's segmentation APIs so you can pinpoint which offerings drive revenue. "
            "Set dimension='product' or 'geographic' and pick period='annual' or 'quarter'. "
            "Example: symbol='AAPL', period='annual', dimension='geographic' to get regional "
            "revenue tables for Apple."
        ),
        func=_wrap_tool_func(_run),
        args_schema=SegmentArgs,
    )


def _build_fmp_footnotes_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class FootnoteArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., MSFT.")
        filing_type: Literal["10-K", "10-Q"] = Field(
            "10-K", description="Filing type to target."
        )
        period: Literal["annual", "quarter", "FY", "Q1", "Q2", "Q3", "Q4"] = Field(
            "annual",
            description="Whether to scan annual ('FY') or a specific quarter (Q1-Q4).",
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
        limit: int = 2,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        target_year = year if year is not None else datetime.now().year
        normalized_period_input = (period or "annual").strip()

        def _coerce_period(value: str) -> str:
            upper = value.strip().upper()
            if upper in {"Q1", "Q2", "Q3", "Q4", "FY"}:
                return upper
            if upper == "QUARTER":
                return "Q1"
            return "FY"

        api_period = _coerce_period(normalized_period_input)
        try:
            payload = client.get(
                "financial-reports-json",
                {
                    "symbol": normalized_symbol,
                    "reportType": filing_type,
                    "period": api_period,
                    "year": target_year,
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
            "Look up detailed 10-K/10-Q footnote tables (commitments, segment notes, etc.) "
            "via Financial Modeling Prep's financial-reports-json endpoint. Specify the filing "
            "type, period (FY or quarter), and optional year to constrain results. Example: "
            "symbol='MSFT', filing_type='10-K', period='FY', year=2024 to retrieve the most "
            "recent annual footnote tables for Microsoft."
        ),
        func=_wrap_tool_func(_run),
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
            le=5,
            description="Number of historical periods to merge (max 5 per FMP income/balance/cash APIs).",
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
                    "income-statement",
                    {"symbol": normalized_symbol, "period": cadence, "limit": limit},
                )
            )
            balance = _ensure_record_list(
                client.get(
                    "balance-sheet-statement",
                    {"symbol": normalized_symbol, "period": cadence, "limit": limit},
                )
            )
            cash = _ensure_record_list(
                client.get(
                    "cash-flow-statement",
                    {"symbol": normalized_symbol, "period": cadence, "limit": limit},
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
            "Pull normalized income, balance-sheet, and cash-flow statements merged by filing "
            "period (FMP caps these endpoints at 5 rows per request). Use when you need a consistent "
            "multi-period view for ratio analysis or trend commentary. Example: symbol='NVDA', "
            "period='quarter', limit=5 to grab the last five quarterly fundamentals."
        ),
        func=_wrap_tool_func(_run),
        args_schema=FundamentalsArgs,
    )


def _build_fmp_ratios_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class RatioArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., GOOG.")
        period: Literal["annual"] = Field( # annual and quarter but annual is free
            "annual", description="Whether to sample annual or quarterly filings. Currently supports only annual."
        )
        limit: int = Field(
            5,
            ge=1,
            le=5,
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
            ratios = _ensure_record_list(
                client.get("ratios", dict(params, symbol=normalized_symbol))
            )
            metrics = _ensure_record_list(
                client.get("key-metrics", dict(params, symbol=normalized_symbol))
            )
            growth = (
                _ensure_record_list(
                    client.get("financial-growth", dict(params, symbol=normalized_symbol))
                )
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
            "Retrieve valuation/efficiency ratios, key metrics, and (optionally) financial "
            "growth rates from Financial Modeling Prep instead of calculating them yourself. "
            "Example: symbol='GOOG', period='annual', include_growth=True, limit=5 to get five "
            "years of P/E, ROE, EV/EBITDA, and growth metrics."
        ),
        func=_wrap_tool_func(_run),
        args_schema=RatioArgs,
    )


def _build_fmp_balance_sheet_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class BalanceArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., AAPL.")
        period: Literal["annual", "quarter", "FY", "Q1", "Q2", "Q3", "Q4"] = Field(
            "annual",
            description="Reporting cadence (annual/FY or a specific quarter).",
        )
        limit: int = Field(
            5,
            ge=1,
            le=5,
            description="Maximum number of rows (FMP balance-sheet endpoint caps at 5).",
        )

    def _normalize_symbol_input(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty.")
        return cleaned

    def _normalize_period(value: str) -> str:
        upper = (value or "annual").strip().upper()
        if upper in {"ANNUAL", "YEAR", "FY"}:
            return "annual"
        if upper in {"QUARTER", "Q1", "Q2", "Q3", "Q4"}:
            return "quarter"
        return "annual"

    def _run(symbol: str, period: str, limit: int) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        cadence = _normalize_period(period)
        try:
            entries = _ensure_record_list(
                client.get(
                    "balance-sheet-statement",
                    {"symbol": normalized_symbol, "period": cadence, "limit": limit},
                )
            )
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        records = [_strip_nones(entry) for entry in entries[:limit] if isinstance(entry, dict)]
        return {
            "symbol": normalized_symbol,
            "period": period,
            "count": len(records),
            "records": records,
            "source": "Financial Modeling Prep stable/balance-sheet-statement",
        }

    return StructuredTool.from_function(
        name="fmp_balance_sheet",
        description=(
            "Fetch raw balance-sheet statements (assets, liabilities, equity) for a ticker. "
            "FMP limits this endpoint to 5 rows per request. Example: symbol='AAPL', "
            "period='annual', limit=5 to retrieve the last five fiscal-year balance sheets."
        ),
        func=_wrap_tool_func(_run),
        args_schema=BalanceArgs,
    )


def _build_fmp_dividend_adjusted_prices_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class PriceArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., AAPL.")
        from_date: Optional[str] = Field(
            None,
            description="Inclusive start date (YYYY-MM-DD).",
        )
        to_date: Optional[str] = Field(
            None,
            description="Inclusive end date (YYYY-MM-DD).",
        )
        limit: Optional[int] = Field(
            None,
            ge=1,
            le=5000,
            description="Optional cap on returned rows (max 5,000).",
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
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        params: Dict[str, Any] = {"symbol": normalized_symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        if limit is not None:
            params["limit"] = min(limit, 5000)

        try:
            entries = _ensure_record_list(
                client.get("historical-price-eod/dividend-adjusted", params)
            )
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        records: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            records.append(
                _strip_nones(
                    {
                        "symbol": entry.get("symbol") or normalized_symbol,
                        "date": entry.get("date"),
                        "adjOpen": entry.get("adjOpen"),
                        "adjHigh": entry.get("adjHigh"),
                        "adjLow": entry.get("adjLow"),
                        "adjClose": entry.get("adjClose"),
                        "volume": entry.get("volume"),
                    }
                )
            )

        return {
            "symbol": normalized_symbol,
            "from": from_date,
            "to": to_date,
            "count": len(records),
            "records": records,
            "source": "Financial Modeling Prep stable/historical-price-eod/dividend-adjusted",
        }

    return StructuredTool.from_function(
        name="fmp_dividend_adjusted_prices",
        description=(
            "Download dividend-adjusted end-of-day prices (adjOpen/adjClose plus volume) for a ticker "
            "using FMP's dividend-adjusted price chart endpoint. Provide optional from/to dates or a "
            "row limit (max 5,000). Example: symbol='AAPL', from_date='2025-06-10', to_date='2025-09-10' "
            "to analyze the summer 2025 adjusted performance."
        ),
        func=_wrap_tool_func(_run),
        args_schema=PriceArgs,
    )


def _build_fmp_dividends_tool(client: FinancialModelingPrepClient) -> StructuredTool:
    class DividendArgs(BaseModel):
        symbol: str = Field(..., description="Ticker symbol, e.g., AAPL.")
        limit: Optional[int] = Field(
            None,
            ge=1,
            le=1000,
            description="Maximum number of dividend rows (FMP default if omitted).",
        )

    def _normalize_symbol_input(value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Symbol cannot be empty.")
        return cleaned

    def _run(symbol: str, limit: Optional[int] = None) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        params: Dict[str, Any] = {"symbol": normalized_symbol}
        if limit is not None:
            params["limit"] = limit

        try:
            entries = _ensure_record_list(client.get("dividends", params))
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        records: List[Dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            records.append(
                _strip_nones(
                    {
                        "symbol": entry.get("symbol") or normalized_symbol,
                        "date": entry.get("date"),
                        "recordDate": entry.get("recordDate"),
                        "paymentDate": entry.get("paymentDate"),
                        "declarationDate": entry.get("declarationDate"),
                        "adjDividend": entry.get("adjDividend"),
                        "dividend": entry.get("dividend"),
                        "yield": entry.get("yield"),
                        "frequency": entry.get("frequency"),
                    }
                )
            )

        return {
            "symbol": normalized_symbol,
            "count": len(records),
            "records": records,
            "source": "Financial Modeling Prep stable/dividends",
        }

    return StructuredTool.from_function(
        name="fmp_dividends",
        description=(
            "Retrieve upcoming or historical dividend events for a single ticker (record/payment/"
            "declaration dates, dividend and adjDividend values, yield, frequency). Example: "
            "symbol='AAPL', limit=20 to see the latest 20 dividend announcements."
        ),
        func=_wrap_tool_func(_run),
        args_schema=DividendArgs,
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
        limit: int = Field(40, ge=1, le=100, description="Number of filings to return.")
        from_date: Optional[str] = Field(
            None,
            description="Inclusive start date (YYYY-MM-DD). Defaults to last 30 days.",
        )
        to_date: Optional[str] = Field(
            None,
            description="Inclusive end date (YYYY-MM-DD). Defaults to today.",
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
        symbol: Optional[str] = None,
        cik: Optional[str] = None,
        form_type: Optional[str] = None,
        page: int = 0,
        limit: int = 40,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        include_raw: bool = False,
    ) -> Dict[str, Any]:
        try:
            normalized_symbol = _normalize_symbol_input(symbol)
            normalized_cik = _normalize_cik_input(cik)
            if not normalized_symbol and not normalized_cik:
                raise ValueError("Provide at least a symbol or a CIK.")
        except ValueError as exc:
            raise ToolException(str(exc)) from exc

        today = datetime.now(timezone.utc).date()
        default_to = today.isoformat()
        default_from = (today - timedelta(days=30)).isoformat()
        query_from = (from_date or default_from).strip()
        query_to = (to_date or default_to).strip()

        if normalized_symbol:
            endpoint = "sec-filings-search/symbol"
            query_params = {
                "symbol": normalized_symbol,
                "from": query_from,
                "to": query_to,
                "page": page,
                "limit": limit,
            }
        elif normalized_cik:
            endpoint = "sec-filings-search/cik"
            query_params = {
                "cik": normalized_cik,
                "from": query_from,
                "to": query_to,
                "page": page,
                "limit": limit,
            }
        else:
            endpoint = "sec-filings-financials"
            query_params = {
                "from": query_from,
                "to": query_to,
                "page": page,
                "limit": limit,
            }
            if form_type:
                query_params["formType"] = form_type

        try:
            payload = client.get(endpoint, query_params)
        except Exception as exc:  # pragma: no cover
            raise ToolException(str(exc)) from exc

        entries = _ensure_record_list(payload)
        filings: List[Dict[str, Any]] = []
        for entry in entries[:limit]:
            record = _strip_nones(
                {
                    "symbol": entry.get("symbol") or normalized_symbol,
                    "cik": entry.get("cik") or normalized_cik,
                    "form_type": entry.get("formType")
                    or entry.get("type")
                    or entry.get("form"),
                    "filed_date": entry.get("filingDate")
                    or entry.get("fillingDate")
                    or entry.get("filedDate"),
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
            "source": f"Financial Modeling Prep stable/{endpoint}",
        }

    return StructuredTool.from_function(
        name="fmp_structured_sec",
        description=(
            "Search recent SEC filings without scraping EDGAR manually. Provide either a ticker "
            "symbol or a CIK (both are optional because the API has dedicated endpoints for each); "
            "if neither is supplied, the tool falls back to the general financials feed filtered by "
            "form_type and date range. Always set the date window via from_date/to_date when you need "
            "specific periods. Example: symbol='AAPL', form_type='10-K', from_date='2024-01-01', "
            "to_date='2024-12-31' to list Apple’s 10-K filings with links."
        ),
        func=_wrap_tool_func(_run),
        args_schema=SecArgs,
    )


def _ensure_record_list(payload: Any) -> List[Dict[str, Any]]:
    """Coerce heterogeneous API responses into a list of dicts."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "financials", "filings", "reports", "historical"):
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


def _ensure_matplotlib_ready():
    if plt is None:  # type: ignore[name-defined]
        raise RuntimeError(
            "matplotlib is required for charting tools. Install it with 'pip install matplotlib'."
        )
    return plt  # type: ignore[return-value]


def _resolve_chart_output_path(
    workspace_dir: Path, requested: Optional[str], prefix: str
) -> Path:
    charts_dir = workspace_dir / "report" / "figures"
    charts_dir.mkdir(parents=True, exist_ok=True)

    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = charts_dir / candidate
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        candidate = charts_dir / f"{prefix}-{timestamp}.png"

    if candidate.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        candidate = candidate.with_suffix(".png")
    candidate = candidate.resolve()
    try:
        candidate.relative_to(workspace_dir)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError("Output path must reside inside the workspace.") from exc
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate