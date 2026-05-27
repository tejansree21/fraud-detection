"""
models/drift_detector.py
ADWIN (Adaptive Windowing) drift detector for monitoring fraud score distribution.

ADWIN works by maintaining a sliding window of observations and detecting
when the mean of recent observations differs significantly from the overall mean.
When drift is detected, it signals that the underlying data distribution has
changed and the models need retraining.

Reference: Bifet & Gavalda (2007) "Learning from Time-Changing Data with
Adaptive Windowing"
"""

from __future__ import annotations
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("drift_detector")


@dataclass
class DriftEvent:
    """Records a detected drift event."""
    timestamp:          str
    window_size:        int
    mean_before:        float
    mean_after:         float
    drift_magnitude:    float
    n_observations:     int


class ADWINDriftDetector:
    """
    ADWIN drift detector tracking fraud score distribution.

    Detects two types of drift:
      1. Score drift    — fraud scores shifting up or down over time
      2. Flag rate drift — proportion of flagged transactions changing

    When drift is detected:
      - Logs a warning with drift magnitude
      - Sets drift_detected = True
      - Records the event for audit trail
    """

    def __init__(self, delta: float = 0.002, min_window: int = 30):
        """
        delta:      confidence parameter (lower = more sensitive)
        min_window: minimum observations before drift can be detected
        """
        self.delta          = delta
        self.min_window     = min_window
        self.window:        deque[float] = deque()
        self.n_total:       int   = 0
        self.drift_detected: bool  = False
        self.drift_events:  list[DriftEvent] = []
        self._total_sum:    float = 0.0
        self._total_sq_sum: float = 0.0

    def update(self, value: float) -> bool:
        """
        Add a new observation. Returns True if drift detected.
        value should be in [0, 1] (e.g. fraud score or flag rate)
        """
        self.window.append(value)
        self._total_sum    += value
        self._total_sq_sum += value ** 2
        self.n_total       += 1
        self.drift_detected = False

        if len(self.window) < self.min_window:
            return False

        # ADWIN cut detection: find if any split of the window
        # shows significantly different means
        detected = self._check_drift()
        if detected:
            self.drift_detected = True
        return detected

    def _check_drift(self) -> bool:
        """
        Check all possible window splits for distribution change.
        Returns True if drift detected at any split point.
        """
        window_list = list(self.window)
        n           = len(window_list)

        total_sum = sum(window_list)
        total_n   = n

        right_sum = 0.0
        right_n   = 0

        # Check splits from right to left
        for i in range(n - 1, max(n // 2, self.min_window), -1):
            right_sum += window_list[i]
            right_n   += 1
            left_n     = total_n - right_n
            left_sum   = total_sum - right_sum

            if left_n < 1:
                continue

            mean_left  = left_sum  / left_n
            mean_right = right_sum / right_n

            # ADWIN epsilon_cut formula
            harmonic   = (1.0 / left_n) + (1.0 / right_n)
            epsilon_cut = math.sqrt(harmonic * math.log(2.0 / self.delta) / 2.0)

            if abs(mean_left - mean_right) >= epsilon_cut:
                drift_magnitude = abs(mean_left - mean_right)
                self._record_drift(mean_left, mean_right, drift_magnitude, n)

                # Shrink window to the more recent portion
                for _ in range(left_n):
                    old = self.window.popleft()
                    self._total_sum    -= old
                    self._total_sq_sum -= old ** 2

                logger.warning(
                    "DRIFT DETECTED | mean_before=%.4f → mean_after=%.4f | "
                    "magnitude=%.4f | window_size=%d",
                    mean_left, mean_right, drift_magnitude, right_n
                )
                return True

        return False

    def _record_drift(
        self, mean_before: float, mean_after: float,
        magnitude: float, window_size: int
    ) -> None:
        from datetime import datetime
        event = DriftEvent(
            timestamp       = datetime.utcnow().isoformat(),
            window_size     = window_size,
            mean_before     = round(mean_before, 4),
            mean_after      = round(mean_after,  4),
            drift_magnitude = round(magnitude,   4),
            n_observations  = self.n_total,
        )
        self.drift_events.append(event)

    @property
    def mean(self) -> float:
        if not self.window:
            return 0.0
        return self._total_sum / len(self.window)

    @property
    def window_size(self) -> int:
        return len(self.window)

    def status(self) -> dict[str, Any]:
        return {
            "window_size":    self.window_size,
            "n_total":        self.n_total,
            "current_mean":   round(self.mean, 4),
            "drift_detected": self.drift_detected,
            "drift_events":   len(self.drift_events),
            "last_drift":     self.drift_events[-1].timestamp if self.drift_events else None,
        }

    def reset(self) -> None:
        self.window.clear()
        self._total_sum    = 0.0
        self._total_sq_sum = 0.0
        self.drift_detected = False
