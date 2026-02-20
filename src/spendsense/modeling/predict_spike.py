from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

DATA_PATH = Path("data/processed/training_dataset.parquet")
METRICS_PATH = Path("reports/baseline_metrics.csv")
MODEL_PATH = Path("reports/models/baseline_xgb.joblib")
OUT_PATH = Path("reports/predictions_spike.csv")

FEATURES_NUM = [
    "txn_count_30d",
    "active_days_30d",
    "income_total_30d",
    "expense_total_30d",
    "net_total_30d",
    "payment_mode_diversity_30d",
    "category_diversity_30d",
    "median_expense_txn_30d",
    "expense_std_30d",
    "top_category_share_30d",
    "expense_last7",
    "expense_prev7",
    "large_txn_cnt_30d",
    "essentials_expense_30d",
    "expense_growth_7d",
    "expense_growth_ratio_7d",
    "essentials_share_30d",
]
FEATURES_CAT = ["top_category_30d"]


def load_xgb_threshold() -> float:
    metrics = pd.read_csv(METRICS_PATH)
    row = metrics.loc[metrics["model"] == "XGBoost"]
    if row.empty:
        raise ValueError("XGBoost row not found in reports/baseline_metrics.csv")
    return float(row["threshold"].iloc[0])


def main():
    thr = load_xgb_threshold()

    bundle = joblib.load(MODEL_PATH)
    preprocess = bundle["preprocess"]
    model = bundle["model"]

    df = pd.read_parquet(DATA_PATH)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    X_t = preprocess.transform(X)

    proba = model.predict_proba(X_t)[:, 1]
    pred = (proba >= thr).astype(int)

    out = pd.DataFrame(
        {
            "as_of_date": df.get("as_of_date"),
            "user_id": df["user_id"],
            "spike_proba": proba,
            "spike_pred": pred,
            "threshold_used": thr,
        }
    )
    out.to_csv(OUT_PATH, index=False)

    print("✅ Model: XGBoost")
    print("✅ Threshold:", thr)
    print("✅ Predicted spike rate:", float(out["spike_pred"].mean()))
    print("✅ Wrote:", OUT_PATH)


if __name__ == "__main__":
    main()