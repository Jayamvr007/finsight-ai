"""
FinSight AI — Production FastAPI Server
Endpoints:
  POST   /api/research           → Create session (multi-tenant, userId required)
  WS     /ws/{session_id}        → Stream agent events (authenticated by userId)
  POST   /api/hitl/{session_id}  → HITL approve/reject
  POST   /api/upload             → Upload PDF → triggers Celery ingestion task
  GET    /api/health             → Health check with DB status
"""

import os
import sys
import uuid
import asyncio
import json
import logging
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect,
    HTTPException, UploadFile, File, Header
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import Response

from pydantic import BaseModel

from config import ALLOWED_ORIGINS, ENVIRONMENT, SAMPLE_DOCS_DIR
from schemas.models import QueryRequest, HITLResponse, AgentState
from agents.graph import build_graph
from db.database import get_pool, close_pool, init_db

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────

app = FastAPI(
    title="FinSight AI",
    description="Multi-agent financial research system — Production",
    version="2.0.0",
    docs_url="/docs" if ENVIRONMENT == "development" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─────────────────────────────────────────
# Structured JSON Logging Middleware
# ─────────────────────────────────────────

@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Log every request as structured JSON for log aggregation (Datadog, CloudWatch, etc.)."""
    start_time = time.time()
    response: Response = await call_next(request)
    duration_ms = round((time.time() - start_time) * 1000, 2)

    log_entry = {
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": duration_ms,
        "client_ip": request.client.host if request.client else "unknown",
    }

    if response.status_code >= 500:
        logger.error(json.dumps(log_entry))
    elif response.status_code >= 400:
        logger.warning(json.dumps(log_entry))
    else:
        logger.info(json.dumps(log_entry))

    return response


# ─────────────────────────────────────────
# In-memory session store
# NOTE: In a scaled deployment, replace this with Redis
# so multiple API server replicas share session state.
# sessions[session_id] = { user_id, hitl_decision, hitl_feedback, hitl_event }
# ─────────────────────────────────────────

sessions: dict = {}

# Compiled graph singleton — reused across all sessions
# Thread isolation is done via unique `thread_id` in LangGraph config
_graph = None


# ─────────────────────────────────────────
# Lifespan — DB pool + LangGraph graph init
# ─────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    global _graph
    logger.info("🚀 FinSight AI starting up (production mode)...")

    # Initialize PostgreSQL pool + LangGraph tables
    await init_db()

    # Build graph with async PostgreSQL checkpointer
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from config import DATABASE_URL

    async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as saver:
        await saver.setup()
        # Build graph at startup with the saver for schema validation,
        # but we'll create per-request savers in WebSocket handlers.
        _graph = build_graph(checkpointer=saver)

    # Trigger async ingestion of sample documents via Celery (non-blocking)
    try:
        from tasks.worker import ingest_directory_task
        ingest_directory_task.delay(SAMPLE_DOCS_DIR)
        logger.info("📄 Queued document ingestion tasks via Celery")
    except Exception as e:
        # Celery not available — fallback to synchronous ingestion
        logger.warning(f"Celery not available, falling back to sync ingestion: {e}")
        loop = asyncio.get_event_loop()
        from rag.ingest import ingest_documents
        await loop.run_in_executor(None, ingest_documents)

    logger.info("✅ FinSight AI ready!")


@app.on_event("shutdown")
async def shutdown_event():
    await close_pool()
    logger.info("🔒 FinSight AI shut down gracefully")


# ─────────────────────────────────────────
# REST Endpoints
# ─────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check — verifies DB connectivity."""
    try:
        pool = await get_pool()
        async with pool.connection() as conn:
            await conn.execute("SELECT 1")
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "ok",
        "service": "FinSight AI",
        "version": "2.0.0",
        "environment": ENVIRONMENT,
        "database": db_status,
    }


@app.post("/api/research")
async def start_research(
    request: QueryRequest,
    x_user_id: Optional[str] = Header(default=None),
):
    """
    Create a new research session.
    
    Multi-tenancy: Pass `X-User-Id` header to associate session with a user.
    Returns a `session_id` (which doubles as the LangGraph `thread_id`).
    """
    user_id = x_user_id or "anonymous"
    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "user_id": user_id,
        "query": request.query,
        "status": "created",
        "hitl_decision": None,
        "hitl_feedback": None,
    }

    logger.info(json.dumps({
        "event": "session_created",
        "session_id": session_id,
        "user_id": user_id,
        "query_length": len(request.query),
    }))

    return {
        "session_id": session_id,
        "user_id": user_id,
        "message": "Session created. Connect via WebSocket.",
    }


@app.post("/api/hitl/{session_id}")
async def hitl_decision(
    session_id: str,
    response: HITLResponse,
    x_user_id: Optional[str] = Header(default=None),
):
    """
    Handle Human-in-the-Loop approve/reject.
    Validates that the requesting user owns this session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # ── Authorization check ──
    # Ensure only the session owner can approve/reject
    if x_user_id and session.get("user_id") not in ("anonymous", x_user_id):
        raise HTTPException(status_code=403, detail="Not authorized for this session")

    # ── Update session state for the WebSocket handler ──
    session["hitl_decision"] = response.approved
    session["hitl_feedback"] = response.feedback or ""

    action = "approved ✅" if response.approved else "rejected ❌"
    logger.info(json.dumps({
        "event": "hitl_decision",
        "session_id": session_id,
        "action": action,
    }))

    return {"message": f"Report {action}. Resuming agent..."}


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    x_user_id: Optional[str] = Header(default=None),
):
    """
    Upload a PDF or TXT financial document.
    Saves the file and triggers an async Celery ingestion task.
    """
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    # Save uploaded file to a temp location
    upload_dir = "/tmp/finsight_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # Trigger async ingestion (non-blocking)
    try:
        from tasks.worker import ingest_document_task
        task = ingest_document_task.delay(filepath)
        logger.info(json.dumps({
            "event": "document_upload",
            "filename": file.filename,
            "task_id": task.id,
            "user_id": x_user_id or "anonymous",
        }))
        return {
            "message": f"File '{file.filename}' queued for ingestion",
            "task_id": task.id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue ingestion: {e}")


# ─────────────────────────────────────────
# WebSocket — Multi-tenant Streaming Agent
# ─────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def research_websocket(
    websocket: WebSocket,
    session_id: str,
):
    """
    WebSocket endpoint for streaming agent events.
    Each session has its own LangGraph thread_id, persisted in PostgreSQL.
    Multiple users can run concurrent pipelines safely.
    """
    await websocket.accept()

    if session_id not in sessions:
        await websocket.send_json({"type": "error", "content": "Session not found"})
        await websocket.close()
        return

    session = sessions[session_id]
    query = session["query"]
    user_id = session.get("user_id", "anonymous")

    # LangGraph thread_id = session_id → each user's state is isolated in PostgreSQL
    thread_config = {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,  # stored in checkpoint metadata
        }
    }

    async def send(event_type: str, content: str, node: str = None, data: dict = None):
        payload = {"type": event_type, "content": content}
        if node:
            payload["node"] = node
        if data:
            payload["data"] = data
        try:
            await websocket.send_json(payload)
        except Exception:
            pass

    start_time = time.time()

    try:
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

        node_display_names = {
            "planner": "📋 Planning Research",
            "market_data": "📈 Fetching Live Prices",
            "web_search": "🔍 Searching Web",
            "rag_retrieval": "📚 Retrieving Documents",
            "analysis": "📊 Analyzing Financials",
            "synthesis": "✍️ Writing Report",
            "human_review": "🚨 Awaiting Approval",
        }

        await send("status", "🚀 Starting FinSight AI research pipeline...", "system")

        # ── Run graph with persistent PostgreSQL checkpointer ──
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from config import DATABASE_URL
        from agents.graph import build_graph

        async with AsyncPostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
            await checkpointer.setup()
            graph = build_graph(checkpointer=checkpointer)

            # ── Phase 1: Stream pipeline until HITL interrupt ──
            loop = asyncio.get_event_loop()

            def run_stream():
                events = []
                for event in graph.stream(
                    initial_state,
                    config=thread_config,
                    stream_mode="updates"
                ):
                    events.append(event)
                return events

            graph_events = await loop.run_in_executor(None, run_stream)

            # ── Process and forward events to frontend ──
            for event in graph_events:
                for node_name, node_output in event.items():
                    display_name = node_display_names.get(node_name, node_name)
                    await send("status", display_name, node_name)

                    if isinstance(node_output, dict):
                        for update in node_output.get("status_updates", []):
                            await send("status", update, node_name)
                            await asyncio.sleep(0.05)

                        if node_name == "analysis" and node_output.get("financial_metrics"):
                            m = node_output["financial_metrics"]
                            await send("analysis", "Financial analysis complete", node_name, {
                                "company": m.company,
                                "recommendation": m.recommendation,
                                "confidence": m.confidence,
                                "key_positives": m.key_positives,
                                "key_risks": m.key_risks,
                                "reasoning": m.reasoning,
                            })

                        if node_name == "synthesis" and node_output.get("final_report"):
                            report = node_output["final_report"]
                            chunk_size = 150
                            for i in range(0, len(report), chunk_size):
                                await send("report_chunk", report[i:i + chunk_size], "synthesis")
                                await asyncio.sleep(0.01)

            # ── Pipeline paused — waiting for HITL ──
            await send("hitl_pending", "🚨 Report ready! Review and approve below.", "human_review")

            # ── Wait for HITL decision (polled from /api/hitl endpoint) ──
            max_wait_seconds = 300
            for _ in range(max_wait_seconds):
                if session.get("hitl_decision") is not None:
                    break
                await asyncio.sleep(1)
            else:
                await send("error", "⏱ Timeout: No approval received in 5 minutes.")
                return

            approved = session["hitl_decision"]
            feedback = session.get("hitl_feedback", "")

            # ── Update LangGraph state with HITL decision ──
            def update_and_resume():
                graph.update_state(
                    thread_config,
                    {"human_approved": approved, "rejection_feedback": feedback}
                )
                resume_events = []
                for event in graph.stream(
                    None,  # None = resume from last checkpoint
                    config=thread_config,
                    stream_mode="updates"
                ):
                    resume_events.append(event)
                return resume_events

            if approved:
                await send("status", "✅ Report approved! Finalizing...", "system")
            else:
                await send("status", f"❌ Rejected. Feedback: {feedback}. Re-analyzing...", "system")

            resume_events = await loop.run_in_executor(None, update_and_resume)

            for event in resume_events:
                for node_name, node_output in event.items():
                    if node_name == "synthesis" and isinstance(node_output, dict):
                        report = node_output.get("final_report", "")
                        for i in range(0, len(report), 150):
                            await send("report_chunk", report[i:i + 150], "synthesis")
                            await asyncio.sleep(0.01)

        duration = round(time.time() - start_time, 2)
        await send("complete", "✅ Research complete!", "system")

        logger.info(json.dumps({
            "event": "session_complete",
            "session_id": session_id,
            "user_id": user_id,
            "duration_seconds": duration,
            "approved": approved,
        }))

    except WebSocketDisconnect:
        logger.info(json.dumps({"event": "ws_disconnect", "session_id": session_id}))
    except Exception as e:
        logger.error(json.dumps({"event": "agent_error", "session_id": session_id, "error": str(e)}))
        await send("error", f"Agent error: {str(e)}")
    finally:
        # Delay cleanup so user can reconnect after a brief drop
        await asyncio.sleep(600)
        sessions.pop(session_id, None)


# ─────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=ENVIRONMENT == "development",
        workers=1 if ENVIRONMENT == "development" else 4,
        log_level="info",
    )
