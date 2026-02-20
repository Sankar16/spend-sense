from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

DATA_PATH = Path("data/processed/training_dataset.parquet")
REPORT_DIR = Path("reports")
MODEL_DIR = REPORT_DIR / "models"

BEST_MODEL_PATH = REPORT_DIR / "best_model.txt"
BEST_THRESHOLD_PATH = REPORT_DIR / "best_threshold.txt"

# Feature list must match train_baseline.py
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


def load_best_model():
    best_model = BEST_MODEL_PATH.read_text().strip()
    thr = float(BEST_THRESHOLD_PATH.read_text().strip())

    if best_model.lower() == "xgboost":
        bundle = joblib.load(MODEL_DIR / "baseline_xgb.joblib")
        preprocess = bundle["preprocess"]
        model = bundle["model"]
        return best_model, thr, preprocess, model, "bundle"

    elif best_model.lower() == "logreg":
        model = joblib.load(MODEL_DIR / "baseline_logreg.joblib")
        return best_model, thr, None, model, "pipeline"

    elif best_model.lower() == "randomforest":
        model = joblib.load(MODEL_DIR / "baseline_rf.joblib")
        return best_model, thr, None, model, "pipeline"

    else:
        raise ValueError(f"Unknown best model in {BEST_MODEL_PATH}: {best_model}")


def main():
    df = pd.read_parquet(DATA_PATH)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    best_model, thr, preprocess, model, model_kind = load_best_model()

    X = df[FEATURES_NUM + FEATURES_CAT].copy()

    if model_kind == "bundle":
        X_t = preprocess.transform(X)
        proba = model.predict_proba(X_t)[:, 1]
    else:
        # pipeline
        proba = model.predict_proba(X)[:, 1]

    df["spike_proba"] = proba
    df["spike_pred"] = (df["spike_proba"] >= thr).astype(int)

    out_path = REPORT_DIR / "predictions_spike.csv"
    df[["as_of_date", "user_id", "spike_proba", "spike_pred"]].to_csv(out_path, index=False)

    print("✅ Best model:", best_model)
    print("✅ Threshold:", thr)
    print("✅ Predicted spike rate:", df["spike_pred"].mean())
    print("✅ Wrote:", out_path)


if __name__ == "__main__":
    main()