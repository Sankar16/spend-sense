from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("reports/models/baseline_logreg.joblib")
THRESH_PATH = Path("reports/best_threshold_logreg.txt")
DATA_PATH = Path("data/processed/training_dataset.parquet")

def main():
    model = joblib.load(MODEL_PATH)
    thr = float(THRESH_PATH.read_text().strip())

    df = pd.read_parquet(DATA_PATH)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    # same features used in training
    FEATURES_NUM = [
        "txn_count_30d","active_days_30d","income_total_30d","expense_total_30d","net_total_30d",
        "payment_mode_diversity_30d","category_diversity_30d","median_expense_txn_30d","expense_std_30d",
        "top_category_share_30d","expense_last7","expense_prev7","large_txn_cnt_30d","essentials_expense_30d",
        "expense_growth_7d","expense_growth_ratio_7d","essentials_share_30d",
    ]
    FEATURES_CAT = ["top_category_30d"]

    X = df[FEATURES_NUM + FEATURES_CAT]
    proba = model.predict_proba(X)[:, 1]
    df["spike_proba"] = proba
    df["spike_pred"] = (proba >= thr).astype(int)

    out_path = Path("reports/predictions_spike.csv")
    df[["as_of_date","user_id","spike_proba","spike_pred"]].to_csv(out_path, index=False)

    print("✅ Wrote:", out_path)
    print("threshold:", thr)
    print("predicted_spike_rate:", df["spike_pred"].mean())

if __name__ == "__main__":
    main()