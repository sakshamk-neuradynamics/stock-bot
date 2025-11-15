import os

from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from .utils import build_rate_limiter

load_dotenv()

# Shared rate limiter for the stock agent
RATE_LIMITER = build_rate_limiter()

# Model (instantiate a LangChain chat model with rate limiting)
MODEL = init_chat_model(
	model="gpt-5-mini",
	model_provider="openai",
	temperature=0,
	rate_limiter=RATE_LIMITER,
)

# Workspace (runtime artifacts)
WORKSPACE_DIR: Path = Path(__file__).resolve().parent / "workspace"

# MCP servers (configure in your MCP runner; strings/URIs are placeholders)
# Example: ["alphavantage://default", "browser://default"]
MCP_SERVERS: Dict[str, Dict[str, str | List[str]]] = {
    "alphavantage": {
      "command": "uvx",
      "args": ["av-mcp", os.getenv("ALPHA_VANTAGE_API_KEY")],
      "transport": "stdio"
    },
    "tavily": {
      "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.getenv('TAVILY_API_KEY')}",
      "transport": "streamable_http"
    },
#     "browsermcp": {
#       "command": "npx",
#       "args": ["@browsermcp/mcp@latest"],
#       "transport": "stdio"
#     }
}

# Quality gates / thresholds
MIN_YEARS_FUNDAMENTALS: int = 5
MIN_PRIMARY_SOURCES_PER_CLAIM: int = 2
MIN_SECONDARY_SOURCES_PER_CLAIM: int = 2

# Concurrency and cost controls
MAX_PARALLEL_SUBAGENTS: int = 4
