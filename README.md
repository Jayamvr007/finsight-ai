# FinSight AI 🚀
### Agentic Financial Research System — LangGraph + RAG + FastAPI + Next.js

> A multi-agent AI system that researches stocks by combining real-time web search,
> document retrieval (RAG), and structured financial analysis — with human-in-the-loop
> approval before any recommendation is surfaced.

---

## Architecture

```
User Query → FastAPI WebSocket
    → LangGraph StateGraph
        → [Planner] Creates research strategy
        → [Web Search] DuckDuckGo real-time news
        → [RAG Retrieval] ChromaDB semantic search over financial docs
        → [Analysis] Gemini + Pydantic structured output
        → [Synthesis] Full markdown research report
        → [HITL] Human approval required ← interrupt()
    → Streaming response to Next.js dashboard
```

## Tech Stack (100% Free)

| Layer | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash (free tier) |
| Orchestration | LangGraph (StateGraph + HITL) |
| RAG | LangChain + ChromaDB + sentence-transformers |
| Web Search | DuckDuckGo Search (no API key) |
| Backend | FastAPI + WebSockets |
| Frontend | Next.js 14 + Vanilla CSS |
| Observability | LangSmith (free tier, optional) |

---

## Setup Instructions

### Step 1 — Get Your Free API Key

Go to: **https://aistudio.google.com/app/apikey**
- Sign in with Google
- Click "Create API Key"
- Copy the key

### Step 2 — Add API Key to .env

```bash
# Edit the .env file in backend/
nano backend/.env
# Replace: GOOGLE_API_KEY=your_google_gemini_api_key_here
# With:    GOOGLE_API_KEY=AIza...your_actual_key
```

### Step 3 — Run Backend

```bash
cd backend
source venv/bin/activate
cd api
python main.py
# Server starts at: http://localhost:8000
# Documents auto-ingested into ChromaDB on first run
```

### Step 4 — Run Frontend (new terminal)

```bash
cd frontend
npm run dev
# Dashboard opens at: http://localhost:3000
```

---

## Usage

1. Open **http://localhost:3000**
2. Type a research query OR click a Quick Query
3. Click **🚀 Start Research**
4. Watch the **Agent Pipeline** execute in real-time (left sidebar)
5. Read the **streaming report** (right panel)
6. When the **🚨 Human Review modal** appears — approve or request revision
7. Final report is generated ✅

---

## What This Demonstrates (AI Developer Skills)

| Concept | Implementation |
|---|---|
| **LangGraph StateGraph** | `agents/graph.py` — 6 nodes, conditional edges |
| **Multi-agent orchestration** | Planner → Search → RAG → Analysis → Synthesis → HITL |
| **Human-in-the-Loop** | `interrupt_before=["human_review"]` + MemorySaver |
| **Production RAG** | ChromaDB hybrid search, company-specific filtering |
| **Tool Calling** | DuckDuckGo, ChromaDB retrieval, financial calculator |
| **Structured Output** | `llm.with_structured_output(FinancialMetrics)` |
| **Streaming API** | FastAPI WebSocket, token-by-token streaming |
| **Observability** | LangSmith tracing (set `LANGSMITH_API_KEY` in .env) |
| **Pydantic v2** | All agent inputs/outputs typed and validated |

---

## Project Structure

```
finsight-ai/
├── backend/
│   ├── config.py                 # Environment + LangSmith setup
│   ├── schemas/models.py         # AgentState + Pydantic models
│   ├── tools/tools.py            # DuckDuckGo, calculator, retrieval tools
│   ├── rag/
│   │   ├── ingest.py             # ChromaDB document ingestion
│   │   ├── retriever.py          # Semantic search queries
│   │   └── sample_docs/          # 5 Indian company financial docs
│   ├── agents/
│   │   ├── nodes.py              # All 6 agent node functions
│   │   └── graph.py              # LangGraph StateGraph definition
│   └── api/main.py               # FastAPI + WebSocket server
└── frontend/
    ├── app/page.tsx              # Main dashboard (WebSocket client)
    └── app/globals.css           # Dark theme design system
```

---

## Resume Bullets

```
FinSight AI — Agentic Financial Research System
Python, LangGraph, LangChain, FastAPI, ChromaDB, Gemini API, Next.js     2026

• Architected a 6-node LangGraph supervisor-worker system decomposing financial 
  queries into parallel research tasks (web search, RAG retrieval, quantitative 
  analysis), delivering complete research reports in under 20 seconds.

• Implemented production-grade RAG pipeline with ChromaDB semantic search using 
  sentence-transformers embeddings, ingesting 5 financial documents with 
  company-specific metadata filtering for context-accurate LLM grounding.

• Engineered Human-in-the-Loop (HITL) checkpoints using LangGraph's 
  interrupt_before API with MemorySaver checkpointing — agents pause before 
  surfacing any BUY/SELL recommendation until human approval is received.

• Built FastAPI async backend with WebSocket endpoints streaming agent status 
  updates and report tokens in real-time; integrated LangSmith for full agent 
  execution tracing and observability.

• Designed Next.js dashboard with live agent pipeline visualization, real-time 
  streaming markdown report, and HITL approval modal — end-to-end full-stack 
  AI product.
```

---

## ⚠️ Disclaimer

This is an educational portfolio project. Not financial advice.
Always consult a SEBI-registered advisor before investing.
