"""
Nodes for the core principles extracting agent.
"""

from pathlib import Path

import tiktoken
import os
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
# from langchain_docling.loader import ExportType, DoclingLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langgraph.types import Send

from core_principles.state import AgentState
from core_principles.prompts import extract_core_principles_prompt, compile_principles_prompt

def _get_encoder(model_hint: str | None = None):
    """Return a tiktoken encoder, falling back gracefully if unavailable."""
    # Prefer cl100k_base for OpenAI GPT-4o/3.5/4 families
    try:
        if model_hint:
            return tiktoken.encoding_for_model(model_hint)
    except KeyError:
        pass
    try:
        return tiktoken.get_encoding("cl100k_base")
    except KeyError:
        return None


def _count_tokens(text: str, encoder=None) -> int:
    if not text:
        return 0
    if encoder is None:
        encoder = _get_encoder("gpt-4o-mini")
    if encoder is None:
        # Fallback heuristic
        return max(1, len(text) // 4)
    try:
        return len(encoder.encode(text))
    except Exception:  # noqa: BLE001
        return max(1, len(text) // 4)


def _chunk_document(doc: Document, max_tokens: int) -> list[Document]:
    """Split a document's content into multiple chunk documents by tokens.

    Args:
        doc: The original document to split.
        max_tokens: Maximum tokens per chunk (content only, no overhead).

    Returns:
        list[Document]: New documents each within the token limit.
    """
    content = doc.page_content or ""
    if not content:
        return [doc]

    encoder = _get_encoder("gpt-4o-mini")
    words = content.split()
    chunks: list[Document] = []
    current: list[str] = []
    for w in words:
        tentative = (" ".join(current + [w])).strip()
        tkns = _count_tokens(tentative, encoder)
        if tkns > max_tokens and current:
            chunks.append(
                Document(page_content=" ".join(current), metadata=doc.metadata)
            )
            current = [w]
        else:
            current.append(w)
    
    if current:
        chunks.append(Document(page_content=" ".join(current), metadata=doc.metadata))
    return chunks


def ingest_documents(_: AgentState) -> AgentState:
    """
    Ingests documents from the 'temp/' directory and updates the agent state with the loaded documents.
    """

    # Collect files from temp, skipping DOC/DOCX for now due to converter issues
    temp_dir = Path.cwd() / "temp"
    all_paths = [p for p in temp_dir.iterdir() if p.is_file()]
    allowed_paths = [
        p for p in all_paths if p.suffix.lower() not in {".docx", ".doc"}
    ]

    if not allowed_paths:
        return {"documents": []}

    documents: list[Document] = []
    # Bulk load allowed files - Takes too long, only use for complex documents
    # loader = DoclingLoader(
    #     file_path=allowed_paths,
    #     export_type=ExportType.MARKDOWN,
    # )
    for path in allowed_paths:
        loader = PDFPlumberLoader(path)
        docs = loader.load()
        documents.extend(docs)

    return {"documents": documents}


def _build_rate_limiter() -> InMemoryRateLimiter:
    """Build a process-wide rate limiter.

    Default spacing is ~17s between requests to avoid 429 with TPM limits.
    Override by setting env var `OPENAI_MIN_SECONDS_BETWEEN_REQUESTS`.
    """
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

RATE_LIMITER = _build_rate_limiter()

def distribute_documents(state: AgentState, config: RunnableConfig):
    """
    Distributes the ingested documents into batches for parallel processing.
    """
    configurables = config.get("configurables", {})
    # Overall model context is ~128k; keep generous headroom for messages/formatting
    max_batch_tokens = int(configurables.get("max_batch_tokens", 60000))
    overhead_tokens = int(configurables.get("overhead_tokens", 3000))

    documents: list[Document] = state["documents"]

    # First, split any single document that exceeds the per-batch capacity
    content_budget = max(1000, max_batch_tokens - overhead_tokens)
    normalized_docs: list[Document] = []
    for doc in documents:
        if _count_tokens(doc.page_content) > content_budget:
            normalized_docs.extend(_chunk_document(doc, content_budget))
        else:
            normalized_docs.append(doc)

    # Group documents into batches under token budget
    batches: list[list[Document]] = []
    current_batch: list[Document] = []
    current_tokens = 0
    sep_tokens = _count_tokens("\n\n==================\n\n")
    for doc in normalized_docs:
        doc_tokens = _count_tokens(doc.page_content)
        candidate = current_tokens + (sep_tokens if current_batch else 0) + doc_tokens
        if current_batch and candidate > content_budget:
            batches.append(current_batch)
            current_batch = [doc]
            current_tokens = doc_tokens
        else:
            if current_batch:
                current_tokens += sep_tokens
            current_batch.append(doc)
            current_tokens += doc_tokens

    if current_batch:
        batches.append(current_batch)

    if len(batches) == 0:
        raise ValueError("No batches found, please check the documents and try again.")
    
    return [
        Send(
            "extract_core_principles",
            {
                "batch_documents": batch,
                "investor_name": state["investor_name"],
            },
        )
        for batch in batches
    ]


def extract_core_principles(state: AgentState) -> AgentState:
    """
    Extracts core principles from the ingested documents and updates the agent state with the extracted principles.
    """

    state["batch_documents"] = "\n\n==================\n\n".join(
        [doc.page_content for doc in state["batch_documents"]]
    )

    llm = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        config_prefix="extract",
        temperature=0,
        rate_limiter=RATE_LIMITER,
    )

    prompt = extract_core_principles_prompt

    agent = prompt | llm | StrOutputParser()

    response = agent.invoke(state)

    return {"core_principles": [response]}


def compile_principles(state: AgentState) -> AgentState:
    """
    Compiles all extracted core principles into a single list and updates the agent state.
    """

    core_principles = "\n\n".join(state.get("core_principles", []))

    prompt = compile_principles_prompt

    llm = init_chat_model(
        model="gpt-5",
        model_provider="openai",
        config_prefix="compile",
        temperature=0,
        rate_limiter=RATE_LIMITER,
    )

    agent = prompt | llm | StrOutputParser()

    response = agent.invoke(
        {
            "investor_name": state["investor_name"],
            "core_principles": core_principles,
        }
    )

    return {"output": response}
