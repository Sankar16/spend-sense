from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Paths:
    processed_dir: Path = Path("data/processed")
    canonical_path: Path = Path("data/processed/transactions_clean.parquet")
    features_path: Path = Path("data/processed/mart_user_30d_features.csv")
    out_path: Path = Path("data/processed/training_dataset.parquet")


def compute_next_window_label(txns: pd.DataFrame, window_end: pd.Timestamp, horizon_days: int = 30) -> pd.DataFrame:
    """
    Label = total EXPENSE per user in the next horizon_days after window_end.
    If window_end = 2022-12-31:
      label window = 2023-01-01 .. 2023-01-30
    """
    start = window_end + pd.Timedelta(days=1)
    end = window_end + pd.Timedelta(days=horizon_days)

    future = txns[(txns["date"] >= start) & (txns["date"] <= end)].copy()
    future = future[future["direction"] == "Expense"].copy()
    labels = (
        future.groupby("user_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "next_30d_expense_total"})
    )
    labels["label_window_start"] = start.date()
    labels["label_window_end"] = end.date()
    return labels


def add_classification_target(df: pd.DataFrame, positive_quantile: float = 0.80) -> pd.DataFrame:
    """
    Convert regression label -> binary target.
    Top 20% (>= 80th percentile) next_30d spend => 1 else 0
    """
    thresh = df["next_30d_expense_total"].quantile(positive_quantile)
    df["high_spend_next_30d"] = (df["next_30d_expense_total"] >= thresh).astype(int)
    df["high_spend_threshold"] = float(thresh)
    return df


def main() -> None:
    p = Paths()

    feat = pd.read_csv(p.features_path, parse_dates=["window_start", "window_end"])
    unique_ends = feat["window_end"].dt.date.unique()
    if len(unique_ends) != 1:
        raise ValueError(f"Expected exactly 1 window_end for now, found: {unique_ends}")

    window_end = pd.to_datetime(feat["window_end"].iloc[0]).normalize()

    txns = pd.read_parquet(p.canonical_path)
    txns["date"] = pd.to_datetime(txns["date"]).dt.normalize()
    txns["amount"] = pd.to_numeric(txns["amount"], errors="coerce")

    required = {"user_id", "date", "amount", "direction"}
    missing = required - set(txns.columns)
    if missing:
        raise ValueError(f"Canonical transactions missing columns: {missing}")

    labels = compute_next_window_label(txns, window_end=window_end, horizon_days=30)

    df = feat.merge(labels, on="user_id", how="left")
    df["next_30d_expense_total"] = df["next_30d_expense_total"].fillna(0.0)

    df = add_classification_target(df, positive_quantile=0.80)

    df.to_parquet(p.out_path, index=False)

    print("Wrote:", p.out_path)
    print("Rows:", len(df), "| positive_rate:", round(df["high_spend_next_30d"].mean(), 3))
    print("Threshold (80th pct):", df["high_spend_threshold"].iloc[0])
    print("Label window:", df["label_window_start"].iloc[0], "→", df["label_window_end"].iloc[0])


if __name__ == "__main__":
    main()