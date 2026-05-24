from __future__ import annotations
from typing import Annotated, Optional, List, Literal
from typing_extensions import TypedDict
import operator

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────
# Pydantic Models (structured LLM output)
# ─────────────────────────────────────────

class FinancialMetrics(BaseModel):
    """Structured output from the Analysis Agent."""
    company: str = Field(description="Company name")
    sector: str = Field(description="Business sector")
    key_positives: List[str] = Field(description="Top 3 positive factors", max_length=3)
    key_risks: List[str] = Field(description="Top 3 risk factors", max_length=3)
    recommendation: Literal["BUY", "HOLD", "SELL", "INSUFFICIENT_DATA"] = Field(
        description="Investment recommendation based on available data"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in recommendation from 0.0 to 1.0"
    )
    reasoning: str = Field(description="1-2 sentence justification for recommendation")


# ─────────────────────────────────────────
# LangGraph Agent State
# ─────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state across all LangGraph nodes."""
    query: str                                                    # User's original question
    company_name: str                                             # Extracted company name
    stock_ticker: str                                             # Yahoo Finance ticker (e.g. INFY.NS)
    research_plan: str                                            # Planner output
    live_market_data: str                                         # Live stock data from Yahoo Finance
    web_results: Annotated[List[str], operator.add]               # Aggregated web search results
    rag_results: Annotated[List[str], operator.add]               # Aggregated ChromaDB results
    financial_metrics: Optional[FinancialMetrics]                 # Structured analysis output
    report_draft: str                                             # Synthesized report text
    final_report: str                                             # Final formatted markdown report
    human_approved: Optional[bool]                                # HITL decision
    rejection_feedback: str                                       # Feedback if rejected
    retry_count: int                                              # Max 2 retries after rejection
    status_updates: Annotated[List[str], operator.add]            # Live status for streaming
    messages: Annotated[List[BaseMessage], add_messages]          # Conversation history


# ─────────────────────────────────────────
# API Request / Response Models
# ─────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(description="User's research query", min_length=5)
    session_id: Optional[str] = None


class HITLResponse(BaseModel):
    session_id: str
    approved: bool
    feedback: Optional[str] = Field(None, description="Reason for rejection if not approved")


class StreamEvent(BaseModel):
    type: Literal[
        "status",       # Agent status update
        "web_result",   # Web search result chunk
        "rag_result",   # RAG retrieval result
        "analysis",     # Financial metrics
        "report_chunk", # Streaming report token
        "hitl_pending", # Waiting for human approval
        "complete",     # Final completion
        "error"         # Error event
    ]
    node: Optional[str] = None     # Which agent node sent this
    content: str = ""
    data: Optional[dict] = None
