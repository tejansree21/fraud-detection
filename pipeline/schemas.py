"""
pipeline/schemas.py
Pydantic models that flow between agents through Kafka and the orchestrator.
These are the contracts — every agent must consume and produce these shapes.
"""

from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from enum import Enum
import uuid


# ── Enums ──────────────────────────────────────────────────────────────────────

class RecommendedAction(str, Enum):
    BLOCK  = "BLOCK"
    REVIEW = "REVIEW"
    ALLOW  = "ALLOW"

class InvestigatorDecision(str, Enum):
    CONFIRMED_FRAUD = "CONFIRMED_FRAUD"
    FALSE_POSITIVE  = "FALSE_POSITIVE"
    NEEDS_REVIEW    = "NEEDS_REVIEW"


# ── Stage 1: Raw transaction (Kafka input) ─────────────────────────────────────

class Transaction(BaseModel):
    transaction_id:     str       = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id:            str
    amount:             float
    merchant:           str
    merchant_category:  str       = "unknown"
    timestamp:          datetime  = Field(default_factory=datetime.utcnow)
    location:           str       = "unknown"
    device_id:          str       = "unknown"
    ip_address:         str       = "unknown"

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Stage 2: Anomaly Detector output ──────────────────────────────────────────

class AnomalyResult(BaseModel):
    transaction_id:   str
    fraud_score:      float           # 0.0 – 1.0
    is_flagged:       bool
    anomaly_vector:   dict[str, Any]  # explanation features
    detector_version: str = "v0.1"
    latency_ms:       float = 0.0
    timestamp:        datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Stage 3: Context Enricher output ──────────────────────────────────────────

class EnrichedTransaction(BaseModel):
    transaction_id:         str
    fraud_score:            float           # carried forward from anomaly result
    merchant_risk_score:    float = 0.0     # 0 low risk → 1 high risk
    geo_consistency_score:  float = 1.0     # 1 = consistent, 0 = suspicious
    device_match:           bool  = True
    velocity_hour:          int   = 0       # tx count last hour for this user
    velocity_day:           int   = 0       # tx count last 24h for this user
    user_behaviour_score:   float = 0.0     # deviation from user baseline
    enriched_payload:       dict[str, Any] = Field(default_factory=dict)
    latency_ms:             float = 0.0
    timestamp:              datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Stage 4: Case Reporter output ─────────────────────────────────────────────

class CaseReport(BaseModel):
    transaction_id:      str
    report_text:         str
    recommended_action:  RecommendedAction
    confidence_score:    float
    similar_cases:       list[dict[str, Any]] = Field(default_factory=list)
    generated_at:        datetime = Field(default_factory=datetime.utcnow)
    latency_ms:          float = 0.0

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Investigator feedback (closes the loop) ───────────────────────────────────

class InvestigatorFeedback(BaseModel):
    transaction_id:  str
    investigator_id: str
    decision:        InvestigatorDecision
    notes:           str = ""
    decided_at:      datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ── Pipeline event: wraps any stage for Kafka serialisation ───────────────────

class PipelineEvent(BaseModel):
    event_id:    str  = Field(default_factory=lambda: str(uuid.uuid4()))
    stage:       str  # "transaction" | "anomaly" | "enriched" | "report" | "feedback"
    payload:     dict[str, Any]
    created_at:  datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
