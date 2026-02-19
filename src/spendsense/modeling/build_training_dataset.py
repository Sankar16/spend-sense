from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Paths:
    canonical_path: Path = Path("data/processed/transactions_clean.parquet")
    out_path: Path = Path("data/processed/training_dataset.parquet")


def add_classification_target(df: pd.DataFrame, positive_quantile: float = 0.80) -> pd.DataFrame:
    thresh = df["next_30d_expense_total"].quantile(positive_quantile)
    df["high_spend_next_30d"] = (df["next_30d_expense_total"] >= thresh).astype(int)
    df["high_spend_threshold"] = float(thresh)
    return df


def compute_features_for_window(txns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """User features for [start, end] inclusive."""
    w = txns[(txns["date"] >= start) & (txns["date"] <= end)].copy()

    # activity
    txn_count = w.groupby("user_id").size().rename("txn_count_30d")
    active_days = w.groupby("user_id")["date"].nunique().rename("active_days_30d")

    # income/expense/net
    income = (
        w[w["direction"] == "Income"].groupby("user_id")["amount"].sum().rename("income_total_30d")
    )
    expense = (
        w[w["direction"] == "Expense"].groupby("user_id")["amount"].sum().rename("expense_total_30d")
    )
    net = (income.fillna(0) - expense.fillna(0)).rename("net_total_30d")

    # diversity features (only expense side is ok too, but we’ll keep all)
    pay_div = w.groupby("user_id")["payment_mode"].nunique().rename("payment_mode_diversity_30d")
    cat_div = w.groupby("user_id")["category"].nunique().rename("category_diversity_30d")

    # expense stats
    exp_txn = w[w["direction"] == "Expense"].copy()
    med_exp = exp_txn.groupby("user_id")["amount"].median().rename("median_expense_txn_30d")
    std_exp = exp_txn.groupby("user_id")["amount"].std().rename("expense_std_30d")

    # top category + share (expense only)
    if len(exp_txn) > 0:
        cat_sum = exp_txn.groupby(["user_id", "category"])["amount"].sum()
        top_cat = cat_sum.groupby(level=0).idxmax().apply(lambda x: x[1]).rename("top_category_30d")
        top_cat_amt = cat_sum.groupby(level=0).max().rename("top_category_amount_30d")
        total_exp = expense.rename("expense_total_30d")
        top_share = (top_cat_amt / total_exp.replace(0, np.nan)).fillna(0).rename("top_category_share_30d")
    else:
        top_cat = pd.Series(dtype=str, name="top_category_30d")
        top_share = pd.Series(dtype=float, name="top_category_share_30d")

    features = pd.concat(
        [txn_count, active_days, income, expense, net, pay_div, cat_div, med_exp, std_exp, top_cat, top_share],
        axis=1,
    ).reset_index()

    # Fill missing numeric features
    num_cols = [
        "txn_count_30d","active_days_30d","income_total_30d","expense_total_30d","net_total_30d",
        "payment_mode_diversity_30d","category_diversity_30d","median_expense_txn_30d","expense_std_30d",
        "top_category_share_30d",
    ]
    for c in num_cols:
        if c in features.columns:
            features[c] = features[c].fillna(0)

    features["window_start"] = start.date()
    features["window_end"] = end.date()
    return features


def compute_label_next_window(txns: pd.DataFrame, label_start: pd.Timestamp, label_end: pd.Timestamp) -> pd.DataFrame:
    """Next-window label (expense total) for [label_start, label_end] inclusive."""
    fut = txns[(txns["date"] >= label_start) & (txns["date"] <= label_end)].copy()
    fut = fut[fut["direction"] == "Expense"].copy()

    labels = fut.groupby("user_id", as_index=False)["amount"].sum().rename(columns={"amount": "next_30d_expense_total"})
    labels["label_window_start"] = label_start.date()
    labels["label_window_end"] = label_end.date()
    return labels


def main() -> None:
    p = Paths()
    txns = pd.read_parquet(p.canonical_path)

    # types
    txns["date"] = pd.to_datetime(txns["date"]).dt.normalize()
    txns["amount"] = pd.to_numeric(txns["amount"], errors="coerce")
    txns = txns.dropna(subset=["user_id", "date", "amount", "direction"])

    min_date = txns["date"].min()
    max_date = txns["date"].max()

    # We need: 30 days of history + 30 days future
    earliest_asof = min_date + pd.Timedelta(days=29)
    latest_asof = max_date - pd.Timedelta(days=30)

    if earliest_asof > latest_asof:
        raise ValueError(
            f"Not enough date span for rolling windows. Need >= 60 days.\n"
            f"min_date={min_date.date()} max_date={max_date.date()}"
        )

    # weekly anchor dates to avoid too many rows (company trick)
    as_of_dates = pd.date_range(earliest_asof, latest_asof, freq="7D")

    rows = []
    for asof in as_of_dates:
        feat_start = asof - pd.Timedelta(days=29)
        feat_end = asof

        label_start = asof + pd.Timedelta(days=1)
        label_end = asof + pd.Timedelta(days=30)

        feats = compute_features_for_window(txns, feat_start, feat_end)
        labels = compute_label_next_window(txns, label_start, label_end)

        df = feats.merge(labels, on="user_id", how="left")
        df["next_30d_expense_total"] = df["next_30d_expense_total"].fillna(0.0)
        df["as_of_date"] = asof.date()

        rows.append(df)

    out = pd.concat(rows, ignore_index=True)

    # classification target computed globally (simple baseline)
    out = add_classification_target(out, positive_quantile=0.80)

    out.to_parquet(p.out_path, index=False)

    print("✅ Wrote:", p.out_path)
    print("Date range:", min_date.date(), "→", max_date.date())
    print("as_of_dates:", len(as_of_dates), "| rows:", len(out))
    print("positive_rate:", round(out["high_spend_next_30d"].mean(), 3))
    print("threshold:", out["high_spend_threshold"].iloc[0])


if __name__ == "__main__":
    main()