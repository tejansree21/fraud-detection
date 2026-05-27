"""
scripts/download_datasets.py
Downloads fraud datasets for the pipeline.

Datasets:
  1. Kaggle Credit Card Fraud (creditcardfraud.zip) — 284K transactions
  2. PaySim (paysim.zip) — 6.3M synthetic mobile money transactions

Usage:
  python scripts/download_datasets.py

Requirements:
  - Kaggle account + API token (~/.kaggle/kaggle.json)
  - pip install kaggle
"""

import os
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_kaggle(dataset: str, filename: str, dest: Path) -> bool:
    """Download a Kaggle dataset if not already present."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return True

    try:
        import kaggle  # noqa
    except ImportError:
        print("  kaggle package not installed. Run: pip install kaggle")
        return False

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"  Kaggle API token not found at {kaggle_json}")
        print("  Get it from: https://www.kaggle.com/settings → API → Create New Token")
        return False

    print(f"  Downloading {dataset}...")
    os.system(f'kaggle datasets download -d {dataset} -p {DATA_DIR} --unzip')
    return dest.exists()


def generate_paysim_sample(n_rows: int = 100_000) -> Path:
    """
    Generate a PaySim-like synthetic dataset using numpy/pandas.
    Used as fallback if Kaggle is not configured.
    """
    import numpy as np
    import pandas as pd

    dest = DATA_DIR / "paysim_sample.csv"
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest

    print(f"  Generating synthetic PaySim sample ({n_rows:,} rows)...")
    rng = np.random.default_rng(42)

    tx_types   = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    type_weights = [0.35, 0.20, 0.20, 0.15, 0.10]

    n_fraud = int(n_rows * 0.013)  # ~1.3% fraud rate (matches real PaySim)
    is_fraud = np.zeros(n_rows, dtype=int)
    fraud_idx = rng.choice(n_rows, size=n_fraud, replace=False)
    is_fraud[fraud_idx] = 1

    tx_type = rng.choice(tx_types, size=n_rows, p=type_weights)

    # Fraud transactions tend to be larger
    amounts = np.where(
        is_fraud,
        rng.uniform(1000, 50000, n_rows),
        rng.exponential(scale=200, size=n_rows) + 1
    )

    df = pd.DataFrame({
        "step":           rng.integers(1, 744, n_rows),   # hour of simulation
        "type":           tx_type,
        "amount":         amounts.round(2),
        "nameOrig":       [f"C{rng.integers(1e8, 1e9)}" for _ in range(n_rows)],
        "oldbalanceOrg":  rng.uniform(0, 100000, n_rows).round(2),
        "newbalanceOrig": rng.uniform(0, 100000, n_rows).round(2),
        "nameDest":       [f"M{rng.integers(1e8, 1e9)}" for _ in range(n_rows)],
        "oldbalanceDest": rng.uniform(0, 100000, n_rows).round(2),
        "newbalanceDest": rng.uniform(0, 100000, n_rows).round(2),
        "isFraud":        is_fraud,
    })

    df.to_csv(dest, index=False)
    fraud_count = is_fraud.sum()
    print(f"  Generated {n_rows:,} transactions ({fraud_count:,} fraud, {fraud_count/n_rows*100:.1f}%)")
    return dest


def generate_cc_fraud_sample(n_rows: int = 50_000) -> Path:
    """
    Generate a credit-card-fraud-like synthetic dataset.
    Mimics the Kaggle CC Fraud structure (V1-V28 PCA features + Amount + Class).
    """
    import numpy as np
    import pandas as pd

    dest = DATA_DIR / "creditcard_sample.csv"
    if dest.exists():
        print(f"  Already exists: {dest}")
        return dest

    print(f"  Generating synthetic CC Fraud sample ({n_rows:,} rows)...")
    rng = np.random.default_rng(123)

    n_fraud = int(n_rows * 0.00172)  # matches real dataset ratio
    labels  = np.zeros(n_rows, dtype=int)
    labels[rng.choice(n_rows, size=n_fraud, replace=False)] = 1

    # 28 PCA-like features
    features = {}
    for i in range(1, 29):
        fraud_mean  = rng.uniform(-2, 2)
        normal_mean = rng.uniform(-0.5, 0.5)
        vals = np.where(
            labels,
            rng.normal(fraud_mean, 1.5, n_rows),
            rng.normal(normal_mean, 1.0, n_rows)
        )
        features[f"V{i}"] = vals.round(6)

    df = pd.DataFrame(features)
    df["Time"]   = rng.uniform(0, 172792, n_rows).round(0)
    df["Amount"] = np.where(labels, rng.uniform(1, 2000, n_rows), rng.exponential(88, n_rows)).round(2)
    df["Class"]  = labels

    df.to_csv(dest, index=False)
    print(f"  Generated {n_rows:,} transactions ({labels.sum():,} fraud)")
    return dest


if __name__ == "__main__":
    print("=== Fraud Detection Dataset Setup ===\n")

    # Try Kaggle first, fall back to synthetic generation
    print("PaySim dataset:")
    kaggle_ok = download_kaggle(
        dataset  = "ealaxi/paysim1",
        filename = "PS_20174392719_1491204439457_log.csv",
        dest     = DATA_DIR / "PS_20174392719_1491204439457_log.csv"
    )
    if not kaggle_ok:
        generate_paysim_sample(100_000)

    print("\nCredit Card Fraud dataset:")
    kaggle_ok = download_kaggle(
        dataset  = "mlg-ulb/creditcardfraud",
        filename = "creditcard.csv",
        dest     = DATA_DIR / "creditcard.csv"
    )
    if not kaggle_ok:
        generate_cc_fraud_sample(50_000)

    print("\nDatasets ready in ./data/")
    for f in DATA_DIR.iterdir():
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name:50s} {size_mb:6.1f} MB")
