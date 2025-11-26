import pytest

from langchain_core.tools.base import ToolException

from stock_analysis.tools import (
    create_assemble_report_tool,
    _build_fmp_segments_tool,
    _build_fmp_footnotes_tool,
    _build_fmp_fundamentals_tool,
    _build_fmp_ratios_tool,
    _build_fmp_balance_sheet_tool,
    _build_fmp_dividend_adjusted_prices_tool,
    _build_fmp_dividends_tool,
    _build_fmp_sec_tool,
)


class DummyClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path not in self.responses:
            raise AssertionError(f"Unexpected path requested: {path}")
        response = self.responses[path]
        return response() if callable(response) else response


def test_create_assemble_report_tool_builds_report(tmp_path):
    workspace = tmp_path
    section_one = workspace / "intro.md"
    section_two_rel = "sections/outlook.md"
    section_two = workspace / section_two_rel
    section_two.parent.mkdir(parents=True, exist_ok=True)

    section_one.write_text("Intro body\n", encoding="utf-8")
    section_two.write_text("Outlook notes\n", encoding="utf-8")

    tool = create_assemble_report_tool(workspace)
    message = tool.func(
        section_headings=["Intro", " Outlook "],
        section_paths=[str(section_one), section_two_rel],
    )

    report_path = workspace / "report" / "report.md"
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "# Intro" in text
    assert "# Outlook" in text
    assert "Intro body" in text
    assert "Outlook notes" in text
    assert "report/report.md assembled with 2 section(s)" in message


def test_create_assemble_report_tool_validates_lengths(tmp_path):
    tool = create_assemble_report_tool(tmp_path)
    result = tool.func(section_headings=["Intro"], section_paths=[])
    assert "section_headings and section_paths must be the same length" in result


def test_fmp_segments_tool_normalizes_symbol_and_structure():
    responses = {
        "revenue-product-segmentation": [
            {
                "symbol": "AAPL",
                "calendarYear": "2024",
                "period": "FY",
                "data": {"Mac": 100, "iPhone": 200},
            }
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_segments_tool(client)

    result = tool.func(
        symbol=" aapl ",
        period="quarter",
        structure="flat",
        limit=1,
        dimension="product",
    )

    assert result["symbol"] == "AAPL"
    assert result["count"] == 1
    assert result["dimension"] == "product"
    record = result["records"][0]
    assert record["segment_data"] == {"Mac": 100, "iPhone": 200}
    assert client.calls == [
        (
            "revenue-product-segmentation",
            {
                "symbol": "AAPL",
                "period": "quarter",
                "structure": "flat",
            },
        )
    ]


def test_fmp_segments_tool_handles_geographic_dimension():
    responses = {
        "revenue-geographic-segmentation": [
            {
                "symbol": "AAPL",
                "calendarYear": "2024",
                "period": "FY",
                "data": {"Americas": 100, "Europe": 50},
            }
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_segments_tool(client)

    result = tool.func(
        symbol="aapl",
        period="annual",
        structure="hierarchical",
        limit=1,
        dimension="geographic",
    )

    assert result["dimension"] == "geographic"
    record = result["records"][0]
    assert record["segment_data"]["Americas"] == 100
    assert client.calls == [
        (
            "revenue-geographic-segmentation",
            {
                "symbol": "AAPL",
                "period": "annual",
                "structure": "flat",
            },
        )
    ]


def test_fmp_footnotes_tool_filters_by_report_type():
    responses = {
        "financial-reports-json": [
            {
                "symbol": "MSFT",
                "reportType": "10-K",
                "calendarYear": "2024",
                "period": "FY",
                "footnotes": {
                    "Note1": {"title": "Revenue Detail", "rows": [{"col": "value"}]}
                },
            },
            {
                "symbol": "MSFT",
                "reportType": "10-Q",
                "calendarYear": "2024",
                "period": "Q1",
                "footnotes": {},
            },
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_footnotes_tool(client)

    result = tool.func(
        symbol="msft",
        filing_type="10-K",
        period="annual",
        year=2024,
        limit=2,
        include_raw=False,
    )

    assert result["count"] == 1
    filing = result["filings"][0]
    assert filing["symbol"] == "MSFT"
    assert filing["footnote_tables"][0]["label"] == "Revenue Detail"
    assert client.calls[0][0] == "financial-reports-json"
    assert client.calls[0][1]["period"] == "FY"


def test_fmp_footnotes_tool_defaults_limit_when_omitted():
    responses = {
        "financial-reports-json": [
            {"symbol": "MSFT", "reportType": "10-K", "calendarYear": "2023", "period": "FY"}
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_footnotes_tool(client)

    tool.func(symbol="msft", filing_type="10-K", period="annual", year=2025, include_raw=False)

    assert client.calls[0][1]["symbol"] == "MSFT"
    # No limit is passed to the endpoint; default behavior is handled client-side
    assert "limit" not in client.calls[0][1]


def test_fmp_fundamentals_tool_merges_sections():
    responses = {
        "income-statement": [
            {"date": "2024-12-31", "revenue": 100, "grossProfit": 50}
        ],
        "balance-sheet-statement": [
            {"date": "2024-12-31", "totalAssets": 200, "netDebt": 20}
        ],
        "cash-flow-statement": [
            {"date": "2024-12-31", "freeCashFlow": 30, "capitalExpenditure": -10}
        ],
    }
    client = DummyClient(responses)
    tool = _build_fmp_fundamentals_tool(client)

    result = tool.func(symbol="aapl", period="annual", limit=1)

    assert result["symbol"] == "AAPL"
    row = result["records"][0]
    assert row["income_statement"]["revenue"] == 100
    assert row["balance_sheet"]["totalAssets"] == 200
    assert row["cash_flow"]["freeCashFlow"] == 30
    assert len(row["income_statement"]) == 2
    assert client.calls == [
        ("income-statement", {"symbol": "AAPL", "period": "annual", "limit": 1}),
        ("balance-sheet-statement", {"symbol": "AAPL", "period": "annual", "limit": 1}),
        ("cash-flow-statement", {"symbol": "AAPL", "period": "annual", "limit": 1}),
    ]


def test_fmp_ratios_tool_combines_metrics_and_growth():
    responses = {
        "ratios": [
            {"date": "2024-12-31", "priceEarningsRatio": 30, "currentRatio": 1.5}
        ],
        "key-metrics": [
            {"date": "2024-12-31", "marketCap": 1_000, "enterpriseValue": 900}
        ],
        "financial-growth": [
            {"date": "2024-12-31", "revenueGrowth": 0.1, "freeCashFlowGrowth": 0.2}
        ],
    }
    client = DummyClient(responses)
    tool = _build_fmp_ratios_tool(client)

    result = tool.func(symbol="aapl", period="annual", limit=1, include_growth=True)

    row = result["records"][0]
    assert row["ratios"]["priceEarningsRatio"] == 30
    assert row["key_metrics"]["marketCap"] == 1_000
    assert row["growth"]["revenueGrowth"] == 0.1
    assert len(client.calls) == 3


def test_fmp_sec_tool_validates_inputs_and_returns_filings():
    responses = {
        "sec-filings-search/symbol": [
            {
                "symbol": "IBM",
                "formType": "10-K",
                "filingDate": "2024-01-01",
                "acceptedDate": "2024-01-02",
                "finalLink": "https://example.com/ibm-10k",
                "period": "FY2023",
                "cik": "123456",
            }
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_sec_tool(client)

    err = tool.func(
        symbol=None,
        cik=None,
        form_type=None,
        page=0,
        limit=10,
        include_raw=False,
    )
    assert "Provide at least a symbol or a CIK" in err

    result = tool.func(
        symbol="ibm",
        cik=None,
        form_type="10-K",
        page=1,
        limit=5,
        from_date="2024-01-01",
        to_date="2024-03-01",
        include_raw=True,
    )

    assert result["count"] == 1
    filing = result["filings"][0]
    assert filing["form_type"] == "10-K"
    assert filing["report_url"] == "https://example.com/ibm-10k"
    assert "raw" in filing
    assert client.calls == [
        (
            "sec-filings-search/symbol",
            {
                "symbol": "IBM",
                "from": "2024-01-01",
                "to": "2024-03-01",
                "page": 1,
                "limit": 5,
            },
        )
    ]


def test_fmp_dividends_tool_fetches_company_dividends():
    responses = {
        "dividends": [
            {
                "symbol": "AAPL",
                "date": "2025-02-10",
                "recordDate": "2025-02-10",
                "paymentDate": "2025-02-13",
                "declarationDate": "2025-01-30",
                "adjDividend": 0.25,
                "dividend": 0.25,
                "yield": 0.004,
                "frequency": "Quarterly",
            }
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_dividends_tool(client)

    result = tool.func(symbol="aapl", limit=25)

    assert result["symbol"] == "AAPL"
    assert result["count"] == 1
    record = result["records"][0]
    assert record["paymentDate"] == "2025-02-13"
    assert record["dividend"] == 0.25
    assert client.calls == [
        (
            "dividends",
            {
                "symbol": "AAPL",
                "limit": 25,
            },
        )
    ]


def test_fmp_balance_sheet_tool_respects_limit_and_period():
    responses = {
        "balance-sheet-statement": [
            {"symbol": "AAPL", "totalAssets": 100, "period": "FY"},
            {"symbol": "AAPL", "totalAssets": 90, "period": "FY-1"},
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_balance_sheet_tool(client)

    result = tool.func(symbol="aapl", period="quarter", limit=3)
    assert result["symbol"] == "AAPL"
    assert result["count"] == 2
    assert result["records"][0]["totalAssets"] == 100
    assert client.calls == [
        (
            "balance-sheet-statement",
            {"symbol": "AAPL", "period": "quarter", "limit": 3},
        )
    ]


def test_fmp_dividend_adjusted_prices_tool_fetches_series():
    responses = {
        "historical-price-eod/dividend-adjusted": [
            {
                "symbol": "AAPL",
                "date": "2025-02-04",
                "adjOpen": 227.2,
                "adjHigh": 233.13,
                "adjLow": 226.65,
                "adjClose": 232.8,
                "volume": 44489128,
            }
        ]
    }
    client = DummyClient(responses)
    tool = _build_fmp_dividend_adjusted_prices_tool(client)

    result = tool.func(
        symbol="aapl", from_date="2025-06-10", to_date="2025-09-10", limit=6000
    )

    assert result["symbol"] == "AAPL"
    assert result["count"] == 1
    record = result["records"][0]
    assert record["adjClose"] == 232.8
    assert client.calls == [
        (
            "historical-price-eod/dividend-adjusted",
            {
                "symbol": "AAPL",
                "from": "2025-06-10",
                "to": "2025-09-10",
                "limit": 5000,
            },
        )
    ]

