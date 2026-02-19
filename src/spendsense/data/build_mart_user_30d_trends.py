from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

TXN_PATH = Path("data/processed/transactions_clean.parquet")
OUT_PATH = Path("data/processed/mart_user_30d_trends.csv")

ESSENTIALS = {"Rent", "Food", "Utilities", "Education", "Health"}  # tweak later


def build_window_features(txn: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """
    For each user, compute trend features using the last 30 days ending at as_of.
    """
    start_30 = as_of - pd.Timedelta(days=29)
    win = txn[(txn["date"] >= start_30) & (txn["date"] <= as_of)].copy()

    # only expenses for spend-trend features
    win_exp = win[win["direction"] == "Expense"].copy()

    # IMPORTANT: make expense magnitudes positive (canonical may store expense as negative)
    win_exp["amount"] = pd.to_numeric(win_exp["amount"], errors="coerce")
    win_exp = win_exp.dropna(subset=["amount"])
    win_exp["amount"] = win_exp["amount"].abs()

    if win_exp.empty:
        return pd.DataFrame(columns=["as_of_date", "user_id"])

    # bucket day ranges inside the 30-day window
    win_exp["days_ago"] = (as_of - win_exp["date"]).dt.days
    # last 7 days: days_ago 0-6 ; prior 7 days: 7-13
    last7 = win_exp[win_exp["days_ago"].between(0, 6)]
    prev7 = win_exp[win_exp["days_ago"].between(7, 13)]

    # per-user totals
    last7_sum = last7.groupby("user_id")["amount"].sum().rename("expense_last7")
    prev7_sum = prev7.groupby("user_id")["amount"].sum().rename("expense_prev7")

    # large transaction count threshold per window (global within window)
    p95 = win_exp["amount"].quantile(0.95)
    large_cnt = (
        win_exp.assign(is_large=win_exp["amount"] >= p95)
        .groupby("user_id")["is_large"]
        .sum()
        .rename("large_txn_cnt_30d")
    )

    # essentials ratio
    essentials_sum = (
        win_exp[win_exp["category"].isin(ESSENTIALS)]
        .groupby("user_id")["amount"]
        .sum()
        .rename("essentials_expense_30d")
    )
    total_sum = win_exp.groupby("user_id")["amount"].sum().rename("expense_total_30d_check")

    out = pd.concat([last7_sum, prev7_sum, large_cnt, essentials_sum, total_sum], axis=1).fillna(0.0)

    # trend features
    out["expense_growth_7d"] = out["expense_last7"] - out["expense_prev7"]

    eps = 1.0  # avoids divide-by-zero / tiny denominators
    out["expense_growth_ratio_7d"] = (out["expense_last7"] + eps) / (out["expense_prev7"] + eps)

    # optional: cap extreme ratios so one weird user doesn't dominate the model
    out["expense_growth_ratio_7d"] = out["expense_growth_ratio_7d"].clip(0, 10)

    out["essentials_share_30d"] = np.where(
        out["expense_total_30d_check"] > 0,
        out["essentials_expense_30d"] / out["expense_total_30d_check"],
        0.0,
    )

    # optional but good: enforce valid range for a "share"
    out["essentials_share_30d"] = out["essentials_share_30d"].clip(0, 1)

    out = out.reset_index()
    out.insert(0, "as_of_date", as_of.date().isoformat())
    return out


def main():
    txns = pd.read_parquet(TXN_PATH)
    txns["date"] = pd.to_datetime(txns["date"])

    # weekly as_of_dates (same approach as training dataset)
    min_date = txns["date"].min()
    max_date = txns["date"].max()

    # earliest as_of must allow a full 30d lookback
    start_asof = min_date + pd.Timedelta(days=29)
    as_of_dates = pd.date_range(start_asof, max_date, freq="7D")

    parts = []
    for as_of in as_of_dates:
        parts.append(build_window_features(txns, as_of))

    out = pd.concat(parts, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"✅ Wrote: {OUT_PATH} | rows: {len(out)} | as_of_dates: {out['as_of_date'].nunique()}")


if __name__ == "__main__":
    main()