"""
agents/case_reporter.py
Case Reporter Agent — Stage 3 of the fraud detection pipeline.

Phase 4: Augmented with similar case retrieval from vector store.
Groq/Llama generates reports enriched with historical context.
"""

from __future__ import annotations
import os
import json
import logging
from groq import Groq
from pipeline.schemas import (
    Transaction, EnrichedTransaction, CaseReport, RecommendedAction
)
from pipeline.vector_store import FraudVectorStore, CaseRecord
from agents.base_agent import BaseAgent

logger = logging.getLogger("agent.case_reporter")

SYSTEM_PROMPT = """You are a senior fraud analyst at a financial institution.
You receive enriched transaction data and similar historical cases, then produce
concise structured investigation reports for your team.

Your reports must:
1. Summarise the transaction and why it was flagged
2. Assess the risk signals (merchant risk, geo consistency, device match, velocity)
3. Reference any similar historical cases provided
4. Recommend one action: BLOCK, REVIEW, or ALLOW
5. Give a confidence score (0.0-1.0) for your recommendation
6. Be factual, professional, and under 300 words

Always respond with ONLY valid JSON in this exact structure:
{
  "report_text": "...",
  "recommended_action": "BLOCK" | "REVIEW" | "ALLOW",
  "confidence_score": 0.0-1.0,
  "key_risk_factors": ["...", "..."],
  "timeline_summary": "..."
}"""


class CaseReporterAgent(BaseAgent):
    """
    Generates LLM-powered investigation reports via Groq API.
    Phase 4: enriched with similar historical case context.

    Inputs:  (Transaction, EnrichedTransaction)
    Outputs: CaseReport
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
    ):
        super().__init__(name="case_reporter", sla_ms=5000.0)
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY must be set in environment or passed explicitly")
        self.client       = Groq(api_key=api_key)
        self.model        = model
        self.vector_store = FraudVectorStore()
        self.logger.info(
            "CaseReporterAgent initialised — model: %s | vector_store: %d cases",
            model, self.vector_store.size()
        )

    # ── BaseAgent interface ────────────────────────────────────────────────────

    def process(self, payload: tuple[Transaction, EnrichedTransaction]) -> CaseReport:
        tx, enriched = payload

        # Build query record for similarity search
        query_record = CaseRecord(
            case_id            = tx.transaction_id,
            transaction_id     = tx.transaction_id,
            fraud_score        = enriched.fraud_score,
            amount             = tx.amount,
            merchant_category  = tx.merchant_category,
            location           = tx.location,
            recommended_action = "REVIEW",
            merchant_risk      = enriched.merchant_risk_score,
            geo_score          = enriched.geo_consistency_score,
            device_match       = enriched.device_match,
            velocity_hour      = enriched.velocity_hour,
            behaviour_score    = enriched.user_behaviour_score,
        )

        # Find similar historical cases
        similar_cases = self.vector_store.search(query_record, k=3, min_score=0.6)

        # Build prompt with similar case context
        user_prompt  = self._build_prompt(tx, enriched, similar_cases)
        raw_response = self._call_groq(user_prompt)
        report       = self._parse_response(
            tx.transaction_id, raw_response, enriched, similar_cases
        )

        # Store this case in the vector store for future lookups
        query_record.recommended_action = report.recommended_action.value
        self.vector_store.add_case(query_record)

        # Periodically save to disk
        if self.vector_store.size() % 20 == 0:
            self.vector_store.save()

        return report

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as exc:
            self.logger.warning("Groq health check failed: %s", exc)
            return False

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_prompt(
        self,
        tx:           Transaction,
        enriched:     EnrichedTransaction,
        similar_cases: list[dict],
    ) -> str:
        similar_section = ""
        if similar_cases:
            similar_section = "\nSimilar Historical Cases:\n"
            for i, case in enumerate(similar_cases, 1):
                similar_section += (
                    f"  Case {i} (similarity={case['similarity']:.2f}): "
                    f"amount=${case['amount']:.2f}, "
                    f"category={case['merchant_category']}, "
                    f"action={case['recommended_action']}, "
                    f"truth={case.get('ground_truth', 'UNKNOWN')}, "
                    f"fraud_score={case['fraud_score']:.3f}\n"
                )
        else:
            similar_section = "\nNo similar historical cases found.\n"

        return f"""
Transaction under review:
- ID:                  {tx.transaction_id}
- User ID:             {tx.user_id}
- Amount:              ${tx.amount:.2f}
- Merchant:            {tx.merchant} ({tx.merchant_category})
- Timestamp:           {tx.timestamp.isoformat()}
- Location:            {tx.location}
- Device ID:           {tx.device_id}
- IP Address:          {tx.ip_address}

Fraud Detection Signals:
- Fraud Score:         {enriched.fraud_score:.4f}  (0=legit, 1=fraud)
- Merchant Risk:       {enriched.merchant_risk_score:.4f}
- Geo Consistency:     {enriched.geo_consistency_score:.4f}
- Device Match:        {"YES — known device" if enriched.device_match else "NO — unrecognised device"}
- Velocity (1h):       {enriched.velocity_hour} transactions in the last hour
- Velocity (24h):      {enriched.velocity_day} transactions in the last 24 hours
- Behaviour Deviation: {enriched.user_behaviour_score:.4f}

Additional Context:
{json.dumps(enriched.enriched_payload, indent=2)}
{similar_section}
Generate a fraud investigation report for this transaction.
"""

    def _call_groq(self, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model    = self.model,
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature     = 0.1,
            max_tokens      = 600,
            response_format = {"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_response(
        self,
        transaction_id: str,
        raw:            str,
        enriched:       EnrichedTransaction,
        similar_cases:  list[dict],
    ) -> CaseReport:
        try:
            data       = json.loads(raw)
            action_str = data.get("recommended_action", "REVIEW").upper()
            action     = RecommendedAction(action_str)
            confidence = float(data.get("confidence_score", 0.5))

            return CaseReport(
                transaction_id     = transaction_id,
                report_text        = data.get("report_text", raw),
                recommended_action = action,
                confidence_score   = confidence,
                similar_cases      = similar_cases,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            self.logger.error("Failed to parse Groq response: %s", exc)
            return CaseReport(
                transaction_id     = transaction_id,
                report_text        = f"Parse error — raw: {raw[:300]}",
                recommended_action = RecommendedAction.REVIEW,
                confidence_score   = 0.0,
                similar_cases      = similar_cases,
            )
