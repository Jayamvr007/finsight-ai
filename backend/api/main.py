"""
FastAPI Application
- POST /api/research → Start a research session, returns session_id
- WebSocket /ws/{session_id} → Streams agent execution events in real-time
- POST /api/hitl/{session_id} → Human approve/reject the report
- GET /api/health → Health check
"""

import os
import sys
import uuid
import asyncio
import json
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas.models import QueryRequest, HITLResponse, AgentState
from agents.graph import graph
from rag.ingest import ingest_documents

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────

app = FastAPI(
    title="FinSight AI",
    description="Multi-agent financial research system powered by LangGraph + RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store {session_id: {state, websocket}}
sessions: dict = {}


# ─────────────────────────────────────────
# Startup — Ingest Documents
# ─────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Run document ingestion on startup (skips if already done)."""
    print("🚀 FinSight AI starting up...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ingest_documents)
    print("✅ Ready to research!")


# ─────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "FinSight AI"}


@app.post("/api/research")
async def start_research(request: QueryRequest):
    """Create a new research session and return session_id."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "query": request.query,
        "status": "created",
        "final_report": None,
        "metrics": None,
        "human_approved": None,
    }
    return {"session_id": session_id, "message": "Session created. Connect via WebSocket."}


@app.post("/api/hitl/{session_id}")
async def hitl_decision(session_id: str, response: HITLResponse):
    """
    Handle Human-in-the-Loop approval or rejection.
    Resumes the paused LangGraph execution with the human decision.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    thread_config = {"configurable": {"thread_id": session_id}}

    # Update state with human decision and resume graph
    update_state = {
        "human_approved": response.approved,
        "rejection_feedback": response.feedback or ""
    }

    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph.update_state(thread_config, update_state)
        )

        # Signal the WebSocket handler to resume
        session["hitl_decision"] = response.approved
        session["hitl_feedback"] = response.feedback or ""
        session["hitl_event"] = asyncio.Event()
        session["hitl_event"].set()

        action = "approved ✅" if response.approved else "rejected ❌"
        return {"message": f"Report {action}. Resuming agent..."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# WebSocket — Streaming Agent Events
# ─────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def research_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint that streams all agent events in real-time.
    Sends JSON events for each stage of the research pipeline.
    """
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({"type": "error", "content": "Session not found"})
        await websocket.close()
        return

    session = sessions[session_id]
    query = session["query"]
    thread_config = {"configurable": {"thread_id": session_id}}

    async def send_event(event_type: str, content: str, node: str = None, data: dict = None):
        """Helper to send structured events to frontend."""
        payload = {"type": event_type, "content": content}
        if node:
            payload["node"] = node
        if data:
            payload["data"] = data
        try:
            await websocket.send_json(payload)
        except Exception:
            pass

    try:
        # Initial state
        initial_state: AgentState = {
            "query": query,
            "company_name": "",
            "stock_ticker": "",
            "research_plan": "",
            "live_market_data": "",
            "web_results": [],
            "rag_results": [],
            "financial_metrics": None,
            "report_draft": "",
            "final_report": "",
            "human_approved": None,
            "rejection_feedback": "",
            "retry_count": 0,
            "status_updates": [],
            "messages": [],
        }

        await send_event("status", "🚀 Starting FinSight AI research pipeline...", "system")

        # Run graph with streaming — yields events for each node
        node_display_names = {
            "planner": "📋 Planning Research",
            "market_data": "📈 Fetching Live Prices",
            "web_search": "🔍 Searching Web",
            "rag_retrieval": "📚 Retrieving Documents",
            "analysis": "📊 Analyzing Financials",
            "synthesis": "✍️ Writing Report",
            "human_review": "🚨 Awaiting Approval",
        }

        loop = asyncio.get_event_loop()

        # Stream graph execution
        def run_graph_stream():
            events = []
            for event in graph.stream(
                initial_state,
                config=thread_config,
                stream_mode="updates"
            ):
                events.append(event)
            return events

        # Run in executor to not block event loop
        graph_events = await loop.run_in_executor(None, run_graph_stream)

        # Process and send events
        final_state = None
        for event in graph_events:
            for node_name, node_output in event.items():
                display_name = node_display_names.get(node_name, node_name)

                await send_event("status", f"{display_name}", node_name)

                # Send status updates from the node
                if isinstance(node_output, dict):
                    for update in node_output.get("status_updates", []):
                        await send_event("status", update, node_name)
                        await asyncio.sleep(0.1)

                    # Stream financial metrics when analysis is done
                    if node_name == "analysis" and node_output.get("financial_metrics"):
                        metrics = node_output["financial_metrics"]
                        metrics_data = {
                            "company": metrics.company,
                            "recommendation": metrics.recommendation,
                            "confidence": metrics.confidence,
                            "key_positives": metrics.key_positives,
                            "key_risks": metrics.key_risks,
                            "reasoning": metrics.reasoning,
                        }
                        await send_event("analysis", "Financial analysis complete", node_name, metrics_data)

                    # Stream report chunks when synthesis is done
                    if node_name == "synthesis" and node_output.get("final_report"):
                        report = node_output["final_report"]
                        # Stream in chunks for effect
                        chunk_size = 100
                        for i in range(0, len(report), chunk_size):
                            chunk = report[i:i + chunk_size]
                            await send_event("report_chunk", chunk, "synthesis")
                            await asyncio.sleep(0.01)

                final_state = node_output

        # Graph is now paused at human_review (interrupt_before)
        await send_event(
            "hitl_pending",
            "🚨 Report ready! Please review and approve or reject below.",
            "human_review"
        )

        # Wait for HITL decision from /api/hitl endpoint
        max_wait = 300  # 5 minutes timeout
        waited = 0
        while waited < max_wait:
            if session.get("hitl_decision") is not None:
                break
            await asyncio.sleep(1)
            waited += 1

        if session.get("hitl_decision") is None:
            await send_event("error", "Timeout waiting for approval. Session expired.")
            return

        approved = session["hitl_decision"]
        feedback = session.get("hitl_feedback", "")

        if approved:
            await send_event("status", "✅ Report approved! Finalizing...", "system")
        else:
            await send_event("status", f"❌ Report rejected. Feedback: {feedback}. Re-analyzing...", "system")

        # Resume graph after HITL
        def resume_graph():
            events = []
            for event in graph.stream(
                None,  # None = resume from checkpoint
                config=thread_config,
                stream_mode="updates"
            ):
                events.append(event)
            return events

        resume_events = await loop.run_in_executor(None, resume_graph)

        for event in resume_events:
            for node_name, node_output in event.items():
                if node_name == "synthesis" and isinstance(node_output, dict):
                    report = node_output.get("final_report", "")
                    if report:
                        for i in range(0, len(report), 100):
                            await send_event("report_chunk", report[i:i + 100], "synthesis")
                            await asyncio.sleep(0.01)

        await send_event("complete", "✅ Research complete!", "system")

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        await send_event("error", f"Agent error: {str(e)}")
    finally:
        # Cleanup session after some time
        await asyncio.sleep(300)
        sessions.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
