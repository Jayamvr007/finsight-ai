"use client";

import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = "http://localhost:8000";
const WS_BASE = "ws://localhost:8000";

// ─── Types ───────────────────────────────────────────────────────
interface StreamEvent {
  type:
    | "status"
    | "web_result"
    | "rag_result"
    | "analysis"
    | "report_chunk"
    | "hitl_pending"
    | "complete"
    | "error";
  node?: string;
  content: string;
  data?: {
    company?: string;
    recommendation?: "BUY" | "HOLD" | "SELL" | "INSUFFICIENT_DATA";
    confidence?: number;
    key_positives?: string[];
    key_risks?: string[];
    reasoning?: string;
  };
}

interface PipelineStep {
  id: string;
  label: string;
  icon: string;
  status: "idle" | "active" | "done" | "pending";
}

const INITIAL_STEPS: PipelineStep[] = [
  { id: "planner", label: "Planning Research", icon: "📋", status: "idle" },
  { id: "market_data", label: "Live Market Data", icon: "📈", status: "idle" },
  { id: "web_search", label: "Web Search", icon: "🔍", status: "idle" },
  { id: "rag_retrieval", label: "Document Retrieval", icon: "📚", status: "idle" },
  { id: "analysis", label: "Financial Analysis", icon: "📊", status: "idle" },
  { id: "synthesis", label: "Report Writing", icon: "✍️", status: "idle" },
  { id: "human_review", label: "Human Review", icon: "🚨", status: "idle" },
];

const QUICK_QUERIES = [
  "Should I invest in Reliance Industries now?",
  "Analyze TCS stock performance and outlook",
  "Is HDFC Bank a good investment in 2025?",
  "Give me a research report on Infosys",
];

// ─── Simple Markdown Renderer ────────────────────────────────────
function renderMarkdown(text: string): string {
  return text
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>');
}

// ─── Main Page ────────────────────────────────────────────────────
export default function Home() {
  const [query, setQuery] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [steps, setSteps] = useState<PipelineStep[]>(INITIAL_STEPS);
  const [statusLogs, setStatusLogs] = useState<string[]>([]);
  const [reportText, setReportText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [showHITL, setShowHITL] = useState(false);
  const [metrics, setMetrics] = useState<StreamEvent["data"] | null>(null);
  const [feedback, setFeedback] = useState("");
  const [isComplete, setIsComplete] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const reportRef = useRef<HTMLDivElement>(null);
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    if (reportRef.current) {
      reportRef.current.scrollTop = reportRef.current.scrollHeight;
    }
  }, [reportText]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [statusLogs]);

  const updateStep = useCallback((nodeId: string, status: PipelineStep["status"]) => {
    setSteps((prev) =>
      prev.map((s) => {
        if (s.id === nodeId) return { ...s, status };
        if (status === "active") {
          // Mark previous steps as done
          const nodeIndex = INITIAL_STEPS.findIndex((n) => n.id === nodeId);
          const stepIndex = INITIAL_STEPS.findIndex((n) => n.id === s.id);
          if (stepIndex < nodeIndex && s.status === "idle") return { ...s, status: "done" };
        }
        return s;
      })
    );
  }, []);

  const addLog = useCallback((msg: string) => {
    setStatusLogs((prev) => [...prev.slice(-20), msg]);
  }, []);

  const startResearch = async () => {
    if (!query.trim() || isRunning) return;

    // Reset state
    setIsRunning(true);
    setIsComplete(false);
    setReportText("");
    setMetrics(null);
    setStatusLogs([]);
    setShowHITL(false);
    setSteps(INITIAL_STEPS);
    addLog("🚀 Initializing research session...");

    try {
      // 1. Create session
      const res = await fetch(`${API_BASE}/api/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const { session_id } = await res.json();
      setSessionId(session_id);

      // 2. Connect WebSocket
      const ws = new WebSocket(`${WS_BASE}/ws/${session_id}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data: StreamEvent = JSON.parse(event.data);

        switch (data.type) {
          case "status":
            addLog(data.content);
            if (data.node) updateStep(data.node, "active");
            break;

          case "analysis":
            if (data.data) setMetrics(data.data);
            if (data.node) updateStep(data.node, "done");
            break;

          case "report_chunk":
            setIsTyping(true);
            setReportText((prev) => prev + data.content);
            if (data.node) updateStep(data.node, "active");
            break;

          case "hitl_pending":
            setIsTyping(false);
            updateStep("human_review", "pending");
            setShowHITL(true);
            addLog("🚨 Awaiting your approval...");
            break;

          case "complete":
            setIsTyping(false);
            setIsRunning(false);
            setIsComplete(true);
            setSteps((prev) => prev.map((s) => ({ ...s, status: "done" })));
            addLog("✅ Research complete!");
            ws.close();
            break;

          case "error":
            addLog(`❌ Error: ${data.content}`);
            setIsRunning(false);
            setIsTyping(false);
            ws.close();
            break;
        }
      };

      ws.onerror = () => {
        addLog("❌ WebSocket connection error");
        setIsRunning(false);
      };

      ws.onclose = () => {
        setIsRunning(false);
      };
    } catch (err) {
      addLog("❌ Failed to start research session");
      setIsRunning(false);
    }
  };

  const handleHITL = async (approved: boolean) => {
    if (!sessionId) return;
    setShowHITL(false);
    addLog(approved ? "✅ Report approved — finalizing..." : "❌ Report rejected — regenerating...");

    if (!approved) setReportText("");

    try {
      await fetch(`${API_BASE}/api/hitl/${sessionId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          approved,
          feedback: feedback || undefined,
        }),
      });
    } catch {
      addLog("❌ HITL submission failed");
    }
    setFeedback("");
  };

  const getRecClass = (rec?: string) => {
    if (!rec) return "neutral";
    return rec.toLowerCase() === "insufficient_data" ? "neutral" : rec.toLowerCase();
  };

  return (
    <>
      <div className="app-container">
        {/* Header */}
        <header className="header">
          <div className="logo">
            <div className="logo-icon">📈</div>
            <span className="logo-text">FinSight AI</span>
            <span className="logo-badge">AGENTIC</span>
          </div>
          <div className="header-right">
            <div className="status-dot" />
            <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              LangGraph + RAG + Gemini
            </span>
          </div>
        </header>

        {/* Main Layout */}
        <div className="main-layout">
          {/* Left Panel */}
          <div className="left-panel">
            {/* Search Card */}
            <div className="card">
              <div className="card-title">Research Query</div>
              <div className="search-form">
                <textarea
                  id="research-query-input"
                  className="search-input"
                  placeholder="e.g. Should I invest in Reliance Industries given current market conditions?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={isRunning}
                />
                <button
                  id="start-research-btn"
                  className="search-btn"
                  onClick={startResearch}
                  disabled={isRunning || !query.trim()}
                >
                  {isRunning ? (
                    <>
                      <div className="step-spinner" style={{ width: 16, height: 16 }} />
                      Researching...
                    </>
                  ) : (
                    <> 🚀 Start Research</>
                  )}
                </button>
              </div>
            </div>

            {/* Quick Queries */}
            <div className="card">
              <div className="card-title">Quick Queries</div>
              <div className="quick-queries">
                {QUICK_QUERIES.map((q, i) => (
                  <button
                    key={i}
                    className="quick-btn"
                    onClick={() => setQuery(q)}
                    disabled={isRunning}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Agent Pipeline */}
            <div className="card">
              <div className="card-title">Agent Pipeline</div>
              <div className="pipeline">
                {steps.map((step) => (
                  <div key={step.id} className={`pipeline-step ${step.status}`}>
                    {step.status === "active" ? (
                      <div className="step-spinner" />
                    ) : (
                      <span className="step-icon">
                        {step.status === "done"
                          ? "✅"
                          : step.status === "pending"
                          ? "⏳"
                          : step.icon}
                      </span>
                    )}
                    <span>{step.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Status Log */}
            {statusLogs.length > 0 && (
              <div className="card">
                <div className="card-title">Live Log</div>
                <div className="status-log" ref={logRef}>
                  {statusLogs.map((log, i) => (
                    <div key={i} className="log-entry">
                      {log}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Panel */}
          <div className="right-panel">
            {/* Metrics Bar */}
            {metrics && (
              <div className="metrics-bar">
                <div className={`metric-card ${getRecClass(metrics.recommendation)}`}>
                  <div className={`metric-value ${getRecClass(metrics.recommendation)}`}>
                    {metrics.recommendation === "INSUFFICIENT_DATA"
                      ? "N/A"
                      : metrics.recommendation}
                  </div>
                  <div className="metric-label">Recommendation</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value neutral">
                    {Math.round((metrics.confidence || 0) * 100)}%
                  </div>
                  <div className="metric-label">AI Confidence</div>
                </div>
                <div className="metric-card">
                  <div className="metric-value neutral" style={{ fontSize: "1rem" }}>
                    {metrics.company || "—"}
                  </div>
                  <div className="metric-label">Company</div>
                </div>
                <div className="metric-card">
                  <div
                    className="metric-value neutral"
                    style={{ fontSize: "0.9rem", color: "var(--accent-green)" }}
                  >
                    {isComplete ? "Done" : isRunning ? "Running" : "Idle"}
                  </div>
                  <div className="metric-label">Status</div>
                </div>
              </div>
            )}

            {/* Report */}
            <div className="report-container" style={{ flex: 1 }}>
              <div className="report-header">
                <span style={{ fontSize: "0.85rem", fontWeight: 600 }}>
                  📄 Research Report
                </span>
                {isComplete && (
                  <span
                    style={{
                      fontSize: "0.75rem",
                      color: "var(--accent-green)",
                      background: "rgba(16,185,129,0.1)",
                      padding: "0.2rem 0.6rem",
                      borderRadius: "99px",
                    }}
                  >
                    ✅ Complete
                  </span>
                )}
              </div>

              <div className="report-content" ref={reportRef}>
                {reportText ? (
                  <>
                    <div
                      dangerouslySetInnerHTML={{
                        __html: renderMarkdown(reportText),
                      }}
                    />
                    {isTyping && <span className="typing-cursor" />}
                  </>
                ) : (
                  <div className="empty-state">
                    <div className="empty-icon">📊</div>
                    <div className="empty-title">No research yet</div>
                    <div className="empty-subtitle">
                      Enter a query and click &quot;Start Research&quot; to launch the
                      multi-agent pipeline.
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* HITL Modal */}
      {showHITL && (
        <div className="modal-overlay">
          <div className="modal-card">
            <div className="modal-title">🚨 Human-in-the-Loop Review</div>
            <div className="modal-subtitle">
              The AI agents have completed their research. Review the recommendation below and
              approve or request revision.
            </div>

            {metrics && (
              <div className="modal-recommendation">
                <div
                  className={`rec-badge ${getRecClass(metrics.recommendation)}`}
                >
                  {metrics.recommendation === "BUY" && "📈"}
                  {metrics.recommendation === "SELL" && "📉"}
                  {metrics.recommendation === "HOLD" && "⏸️"}
                  {" "}{metrics.recommendation}
                </div>
                <div className="rec-confidence">
                  AI Confidence: {Math.round((metrics.confidence || 0) * 100)}%
                </div>
                <div className="rec-reasoning">{metrics.reasoning}</div>
              </div>
            )}

            <input
              className="feedback-input"
              placeholder="Optional: Add feedback if rejecting (e.g. 'Include more risk analysis')"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
            />

            <div className="modal-actions">
              <button
                id="hitl-approve-btn"
                className="btn-approve"
                onClick={() => handleHITL(true)}
              >
                ✅ Approve Report
              </button>
              <button
                id="hitl-reject-btn"
                className="btn-reject"
                onClick={() => handleHITL(false)}
              >
                ❌ Request Revision
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
