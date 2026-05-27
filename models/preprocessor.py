"""
models/preprocessor.py
Data preprocessing pipeline for fraud detection model training.

Handles:
  - Feature engineering from PaySim and CC Fraud datasets
  - Temporal train/test splits (no data leakage)
  - Scaling and encoding
  - Class imbalance analysis
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

logger = logging.getLogger("preprocessor")

# ── PaySim feature columns ─────────────────────────────────────────────────────
PAYSIM_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "balance_diff_orig",   # engineered
    "balance_diff_dest",   # engineered
    "amount_log",          # engineered
    "zero_balance_orig",   # engineered
    "zero_balance_dest",   # engineered
]

PAYSIM_LABEL = "isFraud"


class FraudPreprocessor:
    """
    Preprocesses raw fraud datasets for model training.

    Strict temporal split: test set is always the LAST 20% of
    transactions by step/time — never a random split — to prevent
    data leakage and ensure realistic evaluation.
    """

    def __init__(self):
        self.scaler       = StandardScaler()
        self.feature_cols = PAYSIM_FEATURES
        self.is_fitted    = False

    # ── Public interface ───────────────────────────────────────────────────────

    def prepare_paysim(
        self,
        csv_path: Path,
        test_size: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load, engineer features, split temporally, and scale.

        Returns: X_train, X_test, y_train, y_test
        """
        logger.info("Loading PaySim from %s", csv_path)
        df = pd.read_csv(csv_path)
        logger.info("Loaded %d rows | fraud=%d (%.3f%%)",
                    len(df), df[PAYSIM_LABEL].sum(),
                    df[PAYSIM_LABEL].mean() * 100)

        df = self._engineer_features(df)

        # Temporal split: sort by step, split at 80th percentile
        df = df.sort_values("step").reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx]
        test_df  = df.iloc[split_idx:]

        logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))
        logger.info("Train fraud rate: %.3f%% | Test fraud rate: %.3f%%",
                    train_df[PAYSIM_LABEL].mean() * 100,
                    test_df[PAYSIM_LABEL].mean() * 100)

        X_train = train_df[self.feature_cols].values.astype(np.float32)
        X_test  = test_df[self.feature_cols].values.astype(np.float32)
        y_train = train_df[PAYSIM_LABEL].values
        y_test  = test_df[PAYSIM_LABEL].values

        # Fit scaler on TRAIN only — never on test
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_test  = self.scaler.transform(X_test).astype(np.float32)
        self.is_fitted = True

        logger.info("Features: %d | Scaler fitted on train set only", len(self.feature_cols))
        return X_train, X_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a raw transaction dataframe for inference."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted — call prepare_paysim() first")
        df = self._engineer_features(df)
        # Fill any missing columns with 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_cols].values.astype(np.float32)
        return self.scaler.transform(X).astype(np.float32)

    def transform_transaction(self, tx_dict: dict) -> np.ndarray:
        """Transform a single transaction dict for real-time inference."""
        df = pd.DataFrame([tx_dict])
        return self.transform(df)

    def save(self, path: Path) -> None:
        joblib.dump({
            "scaler":       self.scaler,
            "feature_cols": self.feature_cols,
            "is_fitted":    self.is_fitted,
        }, path)
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "FraudPreprocessor":
        data = joblib.load(path)
        p = cls()
        p.scaler       = data["scaler"]
        p.feature_cols = data["feature_cols"]
        p.is_fitted    = data["is_fitted"]
        logger.info("Preprocessor loaded from %s", path)
        return p

    # ── Feature engineering ────────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # One-hot encode transaction type
        if "type" in df.columns:
            for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
                df[f"type_{t}"] = (df["type"] == t).astype(float)
        else:
            for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
                df[f"type_{t}"] = 0.0

        # Balance difference features (key fraud signal in PaySim)
        if "oldbalanceOrg" in df.columns and "newbalanceOrig" in df.columns:
            df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        else:
            df["balance_diff_orig"] = 0.0

        if "oldbalanceDest" in df.columns and "newbalanceDest" in df.columns:
            df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
        else:
            df["balance_diff_dest"] = 0.0

        # Log-transform amount (reduces skewness)
        if "amount" in df.columns:
            df["amount_log"] = np.log1p(df["amount"])
        else:
            df["amount_log"] = 0.0

        # Zero-balance flags (strong fraud indicators)
        if "newbalanceOrig" in df.columns:
            df["zero_balance_orig"] = (df["newbalanceOrig"] == 0).astype(float)
        else:
            df["zero_balance_orig"] = 0.0

        if "newbalanceDest" in df.columns:
            df["zero_balance_dest"] = (df["newbalanceDest"] == 0).astype(float)
        else:
            df["zero_balance_dest"] = 0.0

        # Fill missing numeric columns with 0
        for col in ["amount", "oldbalanceOrg", "newbalanceOrig",
                    "oldbalanceDest", "newbalanceDest"]:
            if col not in df.columns:
                df[col] = 0.0

        return df
