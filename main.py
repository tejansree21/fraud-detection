"""
main.py
FastAPI application — exposes the fraud detection pipeline as REST endpoints.

Endpoints:
  POST /transaction          → run a single transaction through the pipeline
  POST /feedback             → submit investigator decision
  GET  /health               → pipeline health check
  GET  /stats                → pipeline statistics
  GET  /report/{tx_id}       → retrieve a case report by transaction ID
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipeline.schemas import Transaction, InvestigatorFeedback, CaseReport
from orchestrator.orchestrator import FraudOrchestrator, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

# In-memory store for reports (Phase 5 → PostgreSQL)
_report_store: dict[str, CaseReport] = {}
_orchestrator: FraudOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _orchestrator
    logger.info("Starting up — initialising orchestrator...")
    _orchestrator = FraudOrchestrator.from_env()
    logger.info("Orchestrator ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title       = "Multi-Agent Fraud Detection API",
    description = "Real-time fraud detection pipeline — MSc AI Dissertation",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Response models ────────────────────────────────────────────────────────────

from pydantic import BaseModel
from typing import Any

class PipelineResponse(BaseModel):
    transaction_id:   str
    was_flagged:      bool
    fraud_score:      float
    action:           str | None
    confidence:       float | None
    report_available: bool
    total_latency_ms: float
    error:            str | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/transaction", response_model=PipelineResponse, tags=["Pipeline"])
def process_transaction(tx: Transaction):
    """
    Submit a transaction to the fraud detection pipeline.
    Returns the fraud score, recommended action, and latency.
    """
    if _orchestrator is None:
        raise HTTPException(503, "Orchestrator not ready")

    result: PipelineResult = _orchestrator.process_transaction(tx)

    if result.report:
        _report_store[tx.transaction_id] = result.report

    return PipelineResponse(
        transaction_id   = tx.transaction_id,
        was_flagged      = result.was_flagged,
        fraud_score      = result.anomaly.fraud_score,
        action           = result.final_action.value if result.final_action else None,
        confidence       = result.report.confidence_score if result.report else None,
        report_available = result.report is not None,
        total_latency_ms = round(result.total_latency_ms, 2),
        error            = result.error,
    )


@app.post("/feedback", tags=["Feedback Loop"])
def record_feedback(feedback: InvestigatorFeedback):
    """Submit an investigator decision for a previously flagged transaction."""
    if _orchestrator is None:
        raise HTTPException(503, "Orchestrator not ready")
    _orchestrator.record_feedback(feedback)
    return {"status": "ok", "transaction_id": feedback.transaction_id}


@app.get("/report/{transaction_id}", response_model=CaseReport, tags=["Reports"])
def get_report(transaction_id: str):
    """Retrieve the case report for a flagged transaction."""
    report = _report_store.get(transaction_id)
    if not report:
        raise HTTPException(404, f"No report found for transaction {transaction_id}")
    return report


@app.get("/health", tags=["System"])
def health():
    """Pipeline health check — returns status of all agents."""
    if _orchestrator is None:
        return {"status": "starting"}
    return _orchestrator.health_check()


@app.get("/stats", tags=["System"])
def stats():
    """Pipeline statistics — processed count, flag rate, feedback loop status."""
    if _orchestrator is None:
        return {}
    return _orchestrator.stats()
