# src/spendsense/modeling/train_baseline.py
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

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping


@dataclass(frozen=True)
class Paths:
    data_path: Path = Path("data/processed/training_dataset.parquet")
    model_dir: Path = Path("reports/models")
    report_dir: Path = Path("reports")


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
    # trend features
    "expense_last7",
    "expense_prev7",
    "large_txn_cnt_30d",
    "essentials_expense_30d",
    "expense_growth_7d",
    "expense_growth_ratio_7d",
    "essentials_share_30d",
]
FEATURES_CAT = ["top_category_30d"]


def time_split_3way(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Strict time split by as_of_date:
      train: earliest 70%
      val:   next 15%
      test:  last 15%
    """
    dates = sorted(pd.to_datetime(df["as_of_date"]).unique())
    n = len(dates)
    train_cut = int(n * train_frac)
    val_cut = int(n * (train_frac + val_frac))

    train_dates = set(dates[:train_cut])
    val_dates = set(dates[train_cut:val_cut])
    test_dates = set(dates[val_cut:])

    train = df[df["as_of_date"].isin(train_dates)].copy()
    val = df[df["as_of_date"].isin(val_dates)].copy()
    test = df[df["as_of_date"].isin(test_dates)].copy()
    return train, val, test


def build_preprocessor(for_tree: bool):
    """
    - For LR: scale numeric features
    - For tree models (RF/XGB): no scaling needed
    """
    if for_tree:
        num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    else:
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
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1))
    best_thresh = thresholds[max(best_idx - 1, 0)] if len(thresholds) else 0.5
    return float(best_thresh), float(f1[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def evaluate_at_threshold(y_true, proba, threshold: float, name: str, extra_prefix: str = ""):
    pred = (proba >= threshold).astype(int)

    roc = roc_auc_score(y_true, proba)
    pr = average_precision_score(y_true, proba)
    acc = accuracy_score(y_true, pred)
    cm = confusion_matrix(y_true, pred)

    print(f"\n=== {name} ({extra_prefix}threshold={threshold:.4f}) ===")
    print("ROC-AUC:", round(roc, 4))
    print("PR-AUC:", round(pr, 4))
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix:\n", cm)
    print("\nReport:\n", classification_report(y_true, pred, digits=4))

    return {"roc_auc": roc, "pr_auc": pr, "accuracy": acc, "threshold": threshold}


def main():
    p = Paths()
    p.model_dir.mkdir(parents=True, exist_ok=True)
    p.report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(p.data_path)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    # TARGET: spend spike label built from spike_ratio using TRAIN-only threshold
    ratio_col = "spike_ratio_next_30d"
    y_col = "spend_spike_next_30d"

    if ratio_col not in df.columns:
        raise ValueError(
            f"Missing column: {ratio_col}. Did you run build_training_dataset.py to compute spike_ratio_next_30d?"
        )

    train_df, val_df, test_df = time_split_3way(df, train_frac=0.7, val_frac=0.15)

    q = 0.80
    label_thr = float(train_df[ratio_col].quantile(q))
    (p.report_dir / "label_threshold_train_q80.txt").write_text(str(label_thr))

    for part_df in (train_df, val_df, test_df):
        part_df[y_col] = (part_df[ratio_col] >= label_thr).astype(int)

    print("label threshold (train-only):", label_thr)
    print("train positive rate:", float(train_df[y_col].mean()))
    print("val positive rate:", float(val_df[y_col].mean()))
    print("test positive rate:", float(test_df[y_col].mean()))

    X_train = train_df[FEATURES_NUM + FEATURES_CAT]
    y_train = train_df[y_col].astype(int)

    X_val = val_df[FEATURES_NUM + FEATURES_CAT]
    y_val = val_df[y_col].astype(int)

    X_test = test_df[FEATURES_NUM + FEATURES_CAT]
    y_test = test_df[y_col].astype(int)

    # imbalance ratio for XGB
    n_pos = int(y_train.sum())
    n_neg = int((1 - y_train).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1))

    results = []

    # ---------------- LogReg ----------------
    pre_lr = build_preprocessor(for_tree=False)
    lr = Pipeline(
        steps=[
            ("preprocess", pre_lr),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    lr.fit(X_train, y_train)

    lr_val_proba = lr.predict_proba(X_val)[:, 1]
    lr_best_thr, lr_best_f1, lr_best_p, lr_best_r = best_threshold_by_f1(y_val, lr_val_proba)
    print("\n=== LogReg (VAL tuning) ===")
    print("Best threshold (VAL F1):", round(lr_best_thr, 4))
    print("Best F1 (class 1, VAL):", round(lr_best_f1, 4))
    print("Precision@best (VAL):", round(lr_best_p, 4), "| Recall@best (VAL):", round(lr_best_r, 4))

    lr_test_proba = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate_at_threshold(y_test, lr_test_proba, lr_best_thr, "LogReg", extra_prefix="TEST ")
    lr_metrics.update({"model": "LogReg", "best_f1_val": lr_best_f1})
    results.append(lr_metrics)

    joblib.dump(lr, p.model_dir / "baseline_logreg.joblib")

    # ---------------- RandomForest ----------------
    pre_rf = build_preprocessor(for_tree=True)
    rf = Pipeline(
        steps=[
            ("preprocess", pre_rf),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=-1,
            )),
        ]
    )
    rf.fit(X_train, y_train)

    rf_val_proba = rf.predict_proba(X_val)[:, 1]
    rf_best_thr, rf_best_f1, rf_best_p, rf_best_r = best_threshold_by_f1(y_val, rf_val_proba)
    print("\n=== RandomForest (VAL tuning) ===")
    print("Best threshold (VAL F1):", round(rf_best_thr, 4))
    print("Best F1 (class 1, VAL):", round(rf_best_f1, 4))
    print("Precision@best (VAL):", round(rf_best_p, 4), "| Recall@best (VAL):", round(rf_best_r, 4))

    rf_test_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_at_threshold(y_test, rf_test_proba, rf_best_thr, "RandomForest", extra_prefix="TEST ")
    rf_metrics.update({"model": "RandomForest", "best_f1_val": rf_best_f1})
    results.append(rf_metrics)

    joblib.dump(rf, p.model_dir / "baseline_rf.joblib")

    # ---------------- XGBoost (manual early stopping, version-safe) ----------------
    pre_xgb = build_preprocessor(for_tree=True)
    X_train_t = pre_xgb.fit_transform(X_train)
    X_val_t = pre_xgb.transform(X_val)
    X_test_t = pre_xgb.transform(X_test)

    # Try a few tree counts and pick the best on VAL PR-AUC (or VAL F1)
    candidate_estimators = [200, 400, 800, 1200, 2000]

    best_model = None
    best_n = None
    best_val_pr = -1.0
    best_val_proba = None

    for n_est in candidate_estimators:
        m = XGBClassifier(
            n_estimators=n_est,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
        )
        m.fit(X_train_t, y_train)

        val_proba = m.predict_proba(X_val_t)[:, 1]
        val_pr = average_precision_score(y_val, val_proba)

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_model = m
            best_n = n_est
            best_val_proba = val_proba

    print("\n=== XGBoost (manual tuning) ===")
    print("scale_pos_weight:", round(scale_pos_weight, 4))
    print("best_n_estimators (by VAL PR-AUC):", best_n)
    print("best_val_pr_auc:", round(best_val_pr, 4))

    # Threshold tuning on VAL (F1) using the best model’s VAL probabilities
    xgb_best_thr, xgb_best_f1, xgb_best_p, xgb_best_r = best_threshold_by_f1(y_val, best_val_proba)
    print("Best threshold (VAL F1):", round(xgb_best_thr, 4))
    print("Best F1 (class 1, VAL):", round(xgb_best_f1, 4))
    print("Precision@best (VAL):", round(xgb_best_p, 4), "| Recall@best (VAL):", round(xgb_best_r, 4))

    # Evaluate on TEST
    xgb_test_proba = best_model.predict_proba(X_test_t)[:, 1]
    xgb_metrics = evaluate_at_threshold(y_test, xgb_test_proba, xgb_best_thr, "XGBoost", extra_prefix="TEST ")
    xgb_metrics.update({"model": "XGBoost", "best_f1_val": xgb_best_f1})
    results.append(xgb_metrics)

    # Save bundle: preprocessor + chosen model
    xgb_bundle = {"preprocess": pre_xgb, "model": best_model, "best_n_estimators": best_n}
    joblib.dump(xgb_bundle, p.model_dir / "baseline_xgb.joblib")

    # ---------------- Save metrics + pick best by VAL F1 ----------------
    metrics_df = pd.DataFrame(results).sort_values("best_f1_val", ascending=False)
    out_path = p.report_dir / "baseline_metrics.csv"
    metrics_df.to_csv(out_path, index=False)

    print("\n✅ Saved metrics:", out_path)
    print(metrics_df)

    best = metrics_df.iloc[0]
    best_model = str(best["model"])
    best_thr = float(best["threshold"])

    # Save both a pointer file and model-specific threshold file
    (p.report_dir / "best_model.txt").write_text(best_model)
    (p.report_dir / "best_threshold.txt").write_text(str(best_thr))

    thr_path = p.report_dir / f"best_threshold_{best_model.lower()}.txt"
    thr_path.write_text(str(best_thr))

    print(f"✅ Saved best_model.txt + best_threshold.txt: {best_model} @ {best_thr:.4f}")
    print(f"✅ Saved threshold: {thr_path} (model={best_model}, thr={best_thr:.4f})")


if __name__ == "__main__":
    main()