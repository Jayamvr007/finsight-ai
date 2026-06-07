"""
LangGraph StateGraph Definition — Production Version
Compiled per-request with an async PostgreSQL checkpointer for durable state persistence.

Graph Flow:
START → planner → market_data → web_search → rag_retrieval → analysis → synthesis → human_review → END
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
from langgraph.checkpoint.base import BaseCheckpointSaver


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
        return "end"


def increment_retry(state: AgentState) -> dict:
    """Small node to increment retry count before re-synthesis."""
    return {"retry_count": state.get("retry_count", 0) + 1}


# ─────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────

def build_graph(checkpointer: BaseCheckpointSaver):
    """
    Build and compile the LangGraph StateGraph with an injected checkpointer.
    
    Called once per startup with the AsyncPostgresSaver for persistent state.
    The same compiled graph is reused across all sessions — thread isolation
    is handled by the unique `thread_id` in each request's config.
    
    Args:
        checkpointer: Any LangGraph-compatible saver (MemorySaver in dev,
                      AsyncPostgresSaver in production).
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ──
    workflow.add_node("planner", planner_node)
    workflow.add_node("market_data", market_data_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("rag_retrieval", rag_retrieval_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("increment_retry", increment_retry)

    # ── Sequential pipeline edges ──
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "market_data")
    workflow.add_edge("market_data", "web_search")
    workflow.add_edge("web_search", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis", "human_review")

    # ── HITL conditional routing ──
    workflow.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "end": END,
            "retry_synthesis": "increment_retry"
        }
    )

    # ── Retry loop ──
    workflow.add_edge("increment_retry", "synthesis")

    # ── Compile with persistent checkpointer + HITL interrupt ──
    graph = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"]
    )

    return graph
