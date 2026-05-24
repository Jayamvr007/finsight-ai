"""
LangGraph Agent Node Functions
Each function = one node in the StateGraph.
Flow: planner → web_search → rag_retrieval → analysis → synthesis → human_review
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GOOGLE_API_KEY, GEMINI_MODEL
from schemas.models import AgentState, FinancialMetrics
from tools.tools import web_search, retrieve_financial_docs, calculate_financial_metrics
from tools.market_data import get_live_stock_data

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate


# ─────────────────────────────────────────
# Initialize Gemini LLM with Fallback Chain
# ─────────────────────────────────────────

primary_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# Use primary with automatic fallback if it fails
llm = primary_llm.with_fallbacks([fallback_llm])

# Structured output LLM for analysis node
structured_llm = primary_llm.with_structured_output(FinancialMetrics)


# ─────────────────────────────────────────
# Node 1: Planner
# ─────────────────────────────────────────

def planner_node(state: AgentState) -> dict:
    """
    Analyzes the user query and creates a research plan.
    Extracts company name and defines search strategy.
    """
    query = state["query"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial research planner. Extract the company name
        from the query and create a concise research plan.
        Format your response as:
        COMPANY: <exact company name>
        PLAN: <2-3 bullet points of research approach>"""),
        ("human", "Query: {query}")
    ])

    response = llm.invoke(prompt.format_messages(query=query))
    content = response.content

    # Extract company name
    company_name = query  # fallback
    for line in content.split("\n"):
        if line.startswith("COMPANY:"):
            company_name = line.replace("COMPANY:", "").strip()
            break

    # Extract plan
    plan_text = content
    plan_start = content.find("PLAN:")
    if plan_start != -1:
        plan_text = content[plan_start + 5:].strip()

    # Resolve stock ticker for live data
    from tools.market_data import resolve_ticker
    ticker = resolve_ticker(company_name)

    return {
        "company_name": company_name,
        "stock_ticker": ticker,
        "research_plan": plan_text,
        "status_updates": [f"📋 Research plan created for: **{company_name}**"],
        "messages": [HumanMessage(content=query), AIMessage(content=content)]
    }


# ─────────────────────────────────────────
# Node 2: Live Market Data (Yahoo Finance)
# ─────────────────────────────────────────

def market_data_node(state: AgentState) -> dict:
    """
    Fetches real-time stock price, valuation metrics, and fundamentals
    from Yahoo Finance (free, no API key needed).
    """
    company = state["company_name"]

    result = get_live_stock_data.invoke({"company_name": company})

    return {
        "live_market_data": result,
        "status_updates": [f"📈 Live market data fetched for **{company}**"]
    }


# ─────────────────────────────────────────
# Node 3: Web Search
# ─────────────────────────────────────────

def web_search_node(state: AgentState) -> dict:
    """
    Performs real-time web search using DuckDuckGo.
    Searches for latest news, earnings, and analyst opinions.
    """
    company = state["company_name"]
    queries = [
        f"{company} stock analysis 2025 earnings results",
        f"{company} latest news financial performance India",
    ]

    all_results = []
    for q in queries:
        result = web_search.invoke({"query": q})
        if result and "error" not in result.lower():
            all_results.append(f"Search: '{q}'\n{result}")

    combined = "\n\n===\n\n".join(all_results) if all_results else "No web results available."

    return {
        "web_results": [combined],
        "status_updates": [f"🔍 Web search complete for **{company}** — found {len(all_results)} result sets"]
    }


# ─────────────────────────────────────────
# Node 3: RAG Retrieval
# ─────────────────────────────────────────

def rag_retrieval_node(state: AgentState) -> dict:
    """
    Retrieves relevant financial documents from ChromaDB.
    Uses semantic similarity search with company-specific filtering.
    """
    company = state["company_name"]
    query = state["query"]

    result = retrieve_financial_docs.invoke({
        "company_name": company,
        "query": f"{query} financial performance metrics analysis"
    })

    return {
        "rag_results": [result],
        "status_updates": [f"📚 Retrieved internal financial documents for **{company}**"]
    }


# ─────────────────────────────────────────
# Node 4: Financial Analysis
# ─────────────────────────────────────────

def analysis_node(state: AgentState) -> dict:
    """
    Analyzes all gathered data and produces structured output via Pydantic.
    Uses Gemini with structured output for reliable JSON extraction.
    """
    company = state["company_name"]
    web_data = "\n".join(state.get("web_results", []))
    rag_data = "\n".join(state.get("rag_results", []))
    market_data = state.get("live_market_data", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior financial analyst. Based on the research data provided,
        produce a structured financial analysis. Be objective and data-driven.
        If data is insufficient, set recommendation to INSUFFICIENT_DATA.
        Base your confidence on the completeness and quality of available data."""),
        ("human", """Company: {company}

        LIVE MARKET DATA (Yahoo Finance):
        {market_data}

        WEB RESEARCH DATA:
        {web_data}

        INTERNAL DOCUMENT DATA:
        {rag_data}

        Provide a structured analysis with recommendation (BUY/HOLD/SELL/INSUFFICIENT_DATA).""")
    ])

    try:
        metrics: FinancialMetrics = structured_llm.invoke(
            prompt.format_messages(
                company=company,
                market_data=market_data[:2000],
                web_data=web_data[:3000],  # Trim to avoid token limits
                rag_data=rag_data[:3000]
            )
        )
    except Exception:
        # Fallback if structured output fails
        metrics = FinancialMetrics(
            company=company,
            sector="Financial Services",
            key_positives=["Strong market position", "Consistent revenue growth"],
            key_risks=["Market volatility", "Sector headwinds"],
            recommendation="INSUFFICIENT_DATA",
            confidence=0.3,
            reasoning="Insufficient data for a complete analysis. Please check available documents."
        )

    return {
        "financial_metrics": metrics,
        "status_updates": [
            f"📊 Analysis complete — Recommendation: **{metrics.recommendation}** "
            f"(Confidence: {int(metrics.confidence * 100)}%)"
        ]
    }


# ─────────────────────────────────────────
# Node 5: Synthesis
# ─────────────────────────────────────────

def synthesis_node(state: AgentState) -> dict:
    """
    Synthesizes all research into a comprehensive markdown report.
    Incorporates rejection feedback for retry scenarios.
    """
    company = state["company_name"]
    metrics = state.get("financial_metrics")
    market_data = state.get("live_market_data", "")[:2000]
    web_data = "\n".join(state.get("web_results", []))[:2000]
    rag_data = "\n".join(state.get("rag_results", []))[:2000]
    rejection_feedback = state.get("rejection_feedback", "")

    feedback_instruction = ""
    if rejection_feedback:
        feedback_instruction = f"\n\nIMPORTANT: Previous report was rejected. Reviewer feedback: '{rejection_feedback}'. Please address these concerns."

    metrics_summary = ""
    if metrics:
        metrics_summary = f"""
Recommendation: {metrics.recommendation} (Confidence: {int(metrics.confidence * 100)}%)
Key Positives: {', '.join(metrics.key_positives)}
Key Risks: {', '.join(metrics.key_risks)}
Reasoning: {metrics.reasoning}
"""

    prompt = f"""You are a senior financial analyst. Write a comprehensive, professional investment research report in Markdown format.

Company: {company}
Research Plan: {state.get('research_plan', '')}
{feedback_instruction}

STRUCTURED ANALYSIS:
{metrics_summary}

LIVE MARKET DATA:
{market_data}

WEB RESEARCH:
{web_data}

INTERNAL DOCUMENTS:
{rag_data}

Write the report with these sections:
# {company} — Investment Research Report
## Executive Summary
## Business Overview
## Financial Performance
## Key Strengths
## Key Risks
## Market Context
## Investment Recommendation
### ⚠️ Disclaimer
*This is an AI-generated research report for educational and portfolio demonstration purposes only. Not financial advice. Always consult a SEBI-registered financial advisor before investing.*
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content

    return {
        "report_draft": report,
        "final_report": report,
        "status_updates": ["✍️ Research report synthesized — awaiting your approval"]
    }


# ─────────────────────────────────────────
# Node 6: Human-in-the-Loop
# ─────────────────────────────────────────

def human_review_node(state: AgentState) -> dict:
    """
    HITL checkpoint — this node sets hitl_pending flag.
    The actual interrupt() is declared in graph.py via interrupt_before.
    The FastAPI layer handles pausing and resuming the graph.
    """
    return {
        "status_updates": ["🚨 Human approval required — review the report and approve/reject"]
    }
