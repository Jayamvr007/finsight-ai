"""
LangGraph StateGraph Definition
Defines the full agent workflow with HITL interrupt.

Graph Flow:
START → planner → web_search → rag_retrieval → analysis → synthesis → human_review → END
                                                                              ↑
                                                               (interrupt here — waits for human)
                                                               approved → END
                                                               rejected  → synthesis (retry, max 2x)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.models import AgentState
from agents.nodes import (
    planner_node,
    market_data_node,
    web_search_node,
    rag_retrieval_node,
    analysis_node,
    synthesis_node,
    human_review_node,
)

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ─────────────────────────────────────────
# Routing Logic
# ─────────────────────────────────────────

def route_after_human_review(state: AgentState) -> str:
    """
    Conditional edge after human_review node.
    - If approved → END
    - If rejected AND retries remaining → back to synthesis
    - If rejected AND max retries reached → END anyway
    """
    approved = state.get("human_approved")
    retry_count = state.get("retry_count", 0)

    if approved is True:
        return "end"
    elif approved is False and retry_count < 2:
        return "retry_synthesis"
    else:
        # Max retries or no decision yet — end
        return "end"


def increment_retry(state: AgentState) -> dict:
    """Small node to increment retry count before re-synthesis."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ─────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph StateGraph."""

    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("market_data", market_data_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("increment_retry", increment_retry)

    # Define edges (sequential flow)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "market_data")
    workflow.add_edge("market_data", "web_search")
    workflow.add_edge("web_search", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis", "human_review")

    # Conditional edge after human review
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "end": END,
            "retry_synthesis": "increment_retry"
        }
    )

    # After retry increment → back to synthesis
    workflow.add_edge("increment_retry", "synthesis")

    # Memory checkpointer — enables HITL (interrupt + resume)
    memory = MemorySaver()

    # Compile with interrupt BEFORE human_review
    # This pauses execution at human_review and waits for external input
    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"]
    )

    return graph


# Singleton graph instance
graph = build_graph()
