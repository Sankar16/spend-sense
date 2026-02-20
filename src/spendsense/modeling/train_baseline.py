from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import precision_recall_curve



@dataclass(frozen=True)
class Paths:
    data_path: Path = Path("data/processed/training_dataset.parquet")
    model_dir: Path = Path("reports/models")
    report_dir: Path = Path("reports")


FEATURES_NUM = [
    # existing
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

    # NEW trend features (from mart_user_30d_trends)
    "expense_last7",
    "expense_prev7",
    "large_txn_cnt_30d",
    "essentials_expense_30d",
    "expense_growth_7d",
    "expense_growth_ratio_7d",
    "essentials_share_30d",
]
FEATURES_CAT = ["top_category_30d"]


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    dates = sorted(df["as_of_date"].unique())
    cut = int(len(dates) * (1 - test_frac))
    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    train = df[df["as_of_date"].isin(train_dates)].copy()
    test = df[df["as_of_date"].isin(test_dates)].copy()
    return train, test


def build_preprocessor():
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, FEATURES_NUM),
            ("cat", cat_pipe, FEATURES_CAT),
        ]
    )

def best_threshold_by_f1(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall arrays are len(thresholds)+1
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = np.argmax(f1)
    # thresholds is shorter by 1, so handle edge
    best_thresh = thresholds[max(best_idx - 1, 0)] if len(thresholds) else 0.5
    return float(best_thresh), float(f1[best_idx]), float(precision[best_idx]), float(recall[best_idx])

def evaluate(model, X_test, y_test, name: str):
    proba = model.predict_proba(X_test)[:, 1]

    best_thresh, best_f1, best_p, best_r = best_threshold_by_f1(y_test, proba)
    pred = (proba >= best_thresh).astype(int)

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    print(f"\n=== {name} ===")
    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr, 4))
    print("Best threshold (F1):", round(best_thresh, 4))
    print("Best F1 (class 1):", round(best_f1, 4))
    print("Precision@best:", round(best_p, 4), "| Recall@best:", round(best_r, 4))
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(y_test, pred, digits=4))

    return {
        "model": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "accuracy": acc,
        "best_threshold_f1": best_thresh,
        "best_f1_pos": best_f1,
        "precision_pos": best_p,
        "recall_pos": best_r,
    }


def main():
    p = Paths()
    p.model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(p.data_path)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    train_df, test_df = time_split(df, test_frac=0.2)

    X_train = train_df[FEATURES_NUM + FEATURES_CAT]
    # y_train = train_df["high_spend_next_30d"].astype(int)
    y_train = train_df["spend_spike_next_30d"].astype(int)


    X_test = test_df[FEATURES_NUM + FEATURES_CAT]
    # y_test = test_df["high_spend_next_30d"].astype(int)
    y_test  = test_df["spend_spike_next_30d"].astype(int)

    pre = build_preprocessor()

    # 1) Logistic Regression baseline (interpretable)
    lr = Pipeline(
        steps=[
            ("preprocess", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    lr.fit(X_train, y_train)
    lr_metrics = evaluate(lr, X_test, y_test, "LogReg")
    joblib.dump(lr, p.model_dir / "baseline_logreg.joblib")

    # 2) Random Forest baseline (non-linear)
    rf = Pipeline(
        steps=[
            ("preprocess", pre),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )),
        ]
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, "RandomForest")
    joblib.dump(rf, p.model_dir / "baseline_rf.joblib")

    metrics_df = pd.DataFrame([lr_metrics, rf_metrics]).sort_values("pr_auc", ascending=False)
    out_path = p.report_dir / "baseline_metrics.csv"
    metrics_df.to_csv(out_path, index=False)

    print("\n✅ Saved metrics:", out_path)
    print(metrics_df)

    thr_path = p.report_dir / "best_threshold_logreg.txt"
    thr_path.write_text(str(lr_metrics["best_threshold_f1"]))
    print("✅ Saved threshold:", thr_path)


if __name__ == "__main__":
    main()