## Stock KB Deep Agent

This workspace orchestrates a Deep Agent that coordinates LangChain tools, MCP services, and custom analyzers to produce institutional-grade equity research.

### Financial Modeling Prep integration

New LangChain tools pull structured datasets from [Financial Modeling Prep](https://financialmodelingprep.com/) so the agent can:

- Read segment-level revenue/margin mixes (`fmp_segments`)
- Extract structured 10-K/10-Q footnote tables (`fmp_footnote_tables`)
- Merge clean historical fundamentals (`fmp_clean_fundamentals`)
- Retrieve filing-derived ratios & key metrics (`fmp_ratios_metrics`)
- Stream structured SEC filing metadata (`fmp_structured_sec`)

Configure the API credentials in your `.env` (or shell) before launching the agent:

```
FMP_API_KEY=your-key-here
# Optional overrides
FMP_BASE_URL=https://financialmodelingprep.com/api
FMP_HTTP_TIMEOUT=30.0
```

If `FMP_API_KEY` is missing the agent warns and continues without these tools.

### Running the agent

Install dependencies with `uv pip install -r requirements.txt` or `uv pip install .`, set the necessary environment variables (OpenAI, Tavily, Alpha Vantage, FMP), then start the UI or CLI entry points (see `main.py` or `pages/2_Deep_Agent_Chat.py` for Streamlit usage).
