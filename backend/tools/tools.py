"""
Agent Tools
All tools available to LangGraph agents:
- web_search: DuckDuckGo (completely free, no API key)
- retrieve_financial_docs: ChromaDB semantic search
- calculate_metrics: Financial ratio calculator
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import tool
from duckduckgo_search import DDGS
from rag.retriever import retrieve_for_company, retrieve


@tool
def web_search(query: str) -> str:
    """
    Search the web for latest financial news, earnings reports,
    and analyst opinions using DuckDuckGo (free, no API key needed).

    Args:
        query: Search query string (e.g. 'Reliance Industries Q4 2025 results')

    Returns:
        Formatted string of top search results with titles, snippets, and URLs.
    """
    try:
        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5))

        if not search_results:
            return "No web results found for this query."

        for i, r in enumerate(search_results, 1):
            results.append(
                f"[Result {i}]\n"
                f"Title: {r.get('title', 'N/A')}\n"
                f"Summary: {r.get('body', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
            )
        return "\n---\n".join(results)

    except Exception as e:
        return f"Web search error: {str(e)}. Proceeding with available data."


@tool
def retrieve_financial_docs(company_name: str, query: str) -> str:
    """
    Retrieve relevant financial data from the internal document database
    using semantic similarity search (ChromaDB + sentence-transformers).

    Args:
        company_name: Name of the company (e.g. 'Reliance', 'TCS', 'HDFC Bank', 'Infosys')
        query: What information to retrieve (e.g. 'revenue growth and profitability')

    Returns:
        Formatted string of retrieved document chunks with source citations.
    """
    try:
        results = retrieve_for_company(company_name=company_name, query=query, n_results=4)

        if not results:
            return f"No documents found for {company_name}. Using general market data only."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[Doc {i} | Source: {r['source']} | Relevance: {r['relevance_score']}]\n"
                f"{r['text']}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Document retrieval error: {str(e)}"


@tool
def calculate_financial_metrics(
    current_price: float,
    eps: float,
    revenue_current: float,
    revenue_previous: float,
    net_profit: float,
    total_equity: float
) -> str:
    """
    Calculate key financial ratios for investment analysis.

    Args:
        current_price: Current stock price (INR)
        eps: Earnings Per Share (INR)
        revenue_current: Current year revenue (in Crore INR)
        revenue_previous: Previous year revenue (in Crore INR)
        net_profit: Net profit/PAT (in Crore INR)
        total_equity: Total shareholder equity (in Crore INR)

    Returns:
        Formatted string with calculated financial ratios and interpretations.
    """
    try:
        results = {}

        # P/E Ratio
        if eps > 0:
            pe_ratio = round(current_price / eps, 2)
            results["P/E Ratio"] = f"{pe_ratio}x"
            if pe_ratio < 15:
                results["P/E Interpretation"] = "Undervalued (below market avg of ~20x)"
            elif pe_ratio < 25:
                results["P/E Interpretation"] = "Fairly valued"
            else:
                results["P/E Interpretation"] = "Premium valuation — growth expected"

        # Revenue Growth
        if revenue_previous > 0:
            rev_growth = round(((revenue_current - revenue_previous) / revenue_previous) * 100, 2)
            results["Revenue Growth YoY"] = f"{rev_growth}%"
            if rev_growth > 15:
                results["Growth Interpretation"] = "High growth (>15% — strong)"
            elif rev_growth > 8:
                results["Growth Interpretation"] = "Moderate growth (8-15%)"
            else:
                results["Growth Interpretation"] = "Slow growth (<8% — concern)"

        # Net Profit Margin
        if revenue_current > 0:
            npm = round((net_profit / revenue_current) * 100, 2)
            results["Net Profit Margin"] = f"{npm}%"

        # Return on Equity (ROE)
        if total_equity > 0:
            roe = round((net_profit / total_equity) * 100, 2)
            results["Return on Equity (ROE)"] = f"{roe}%"
            if roe > 20:
                results["ROE Interpretation"] = "Excellent capital efficiency (>20%)"
            elif roe > 12:
                results["ROE Interpretation"] = "Good ROE (12-20%)"
            else:
                results["ROE Interpretation"] = "Below average ROE (<12%)"

        output = "=== CALCULATED FINANCIAL METRICS ===\n"
        for key, val in results.items():
            output += f"{key}: {val}\n"

        return output

    except Exception as e:
        return f"Calculation error: {str(e)}"


# Export all tools as a list for LangGraph
ALL_TOOLS = [web_search, retrieve_financial_docs, calculate_financial_metrics]
