"""
Defines the state graph for the core principles extracting agent.
"""

from dotenv import load_dotenv

from langgraph.graph import StateGraph

from core_principles.state import AgentState
from core_principles.nodes import (
    ingest_documents,
    compile_principles,
    extract_core_principles,
    distribute_documents,
)

load_dotenv()

graph = StateGraph(AgentState)

graph.add_node("ingest_documents", ingest_documents)
graph.add_node("extract_core_principles", extract_core_principles)
graph.add_node("compile_principles", compile_principles)

graph.set_entry_point("ingest_documents")
graph.add_conditional_edges(
    "ingest_documents", distribute_documents, ["extract_core_principles"]
)
graph.add_edge("extract_core_principles", "compile_principles")
graph.set_finish_point("compile_principles")

graph = graph.compile()
