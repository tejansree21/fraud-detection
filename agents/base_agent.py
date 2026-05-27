"""
agents/base_agent.py
Abstract base class that every agent in the pipeline must implement.
Enforces a consistent interface: process(), health_check(), and metadata.
"""

from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    """Lightweight metrics tracked per agent instance."""
    total_processed:  int   = 0
    total_errors:     int   = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.total_latency_ms / self.total_processed

    @property
    def error_rate(self) -> float:
        if self.total_processed == 0:
            return 0.0
        return self.total_errors / self.total_processed


class BaseAgent(ABC):
    """
    Abstract base for all fraud-detection pipeline agents.

    Subclasses must implement:
        - process(payload)  → the agent's core logic
        - health_check()    → returns True if the agent is ready

    The base class provides:
        - Timed execution wrapper with SLA enforcement
        - Structured logging
        - Metrics tracking
        - Consistent error handling
    """

    def __init__(self, name: str, sla_ms: float | None = None):
        self.name    = name
        self.sla_ms  = sla_ms          # optional latency budget in milliseconds
        self.metrics = AgentMetrics()
        self.logger  = logging.getLogger(f"agent.{name}")

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, payload: Any) -> Any:
        """
        Timed wrapper around process().
        Handles logging, metrics, SLA checks, and error containment.
        """
        start = time.perf_counter()
        try:
            result = self.process(payload)
            elapsed_ms = (time.perf_counter() - start) * 1000

            self.metrics.total_processed += 1
            self.metrics.total_latency_ms += elapsed_ms

            if self.sla_ms and elapsed_ms > self.sla_ms:
                self.logger.warning(
                    "[%s] SLA breach: %.1fms > %.1fms budget",
                    self.name, elapsed_ms, self.sla_ms
                )
            else:
                self.logger.debug(
                    "[%s] processed in %.1fms", self.name, elapsed_ms
                )

            return result

        except Exception as exc:
            self.metrics.total_errors += 1
            self.logger.error("[%s] error: %s", self.name, exc, exc_info=True)
            raise

    @abstractmethod
    def process(self, payload: Any) -> Any:
        """
        Core agent logic.
        Receives a typed payload, returns a typed result.
        Both types are defined in pipeline/schemas.py.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the agent and its dependencies are ready."""

    # ── Introspection ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "agent":            self.name,
            "healthy":          self.health_check(),
            "total_processed":  self.metrics.total_processed,
            "total_errors":     self.metrics.total_errors,
            "avg_latency_ms":   round(self.metrics.avg_latency_ms, 2),
            "error_rate":       round(self.metrics.error_rate, 4),
            "sla_ms":           self.sla_ms,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
