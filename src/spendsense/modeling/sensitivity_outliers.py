from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class Paths:
    data_path: Path = Path("data/processed/training_dataset.parquet")
    report_dir: Path = Path("reports")
    fig_dir: Path = Path("reports/figures")


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

RATIO_COL = "spike_ratio_next_30d"
Y_COL = "spend_spike_next_30d"


def time_split_3way(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
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


def build_preprocessor():
    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
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
    return float(best_thresh), float(f1[best_idx])


def train_xgb_manual_tuning(X_train_t, y_train, X_val_t, y_val, scale_pos_weight: float):
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

    thr, f1 = best_threshold_by_f1(y_val, best_val_proba)
    return best_model, best_n, best_val_pr, thr, f1


def evaluate(y_true, proba, thr: float):
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "cm": confusion_matrix(y_true, pred),
    }


def build_labels_train_only(train_df, val_df, test_df, q=0.80):
    label_thr = float(train_df[RATIO_COL].quantile(q))
    for part in (train_df, val_df, test_df):
        part[Y_COL] = (part[RATIO_COL] >= label_thr).astype(int)
    return label_thr


def scenario_filters(df: pd.DataFrame):
    # Mean 30d expense per user across as_of_dates
    user_mean_exp = df.groupby("user_id")["expense_total_30d"].mean().sort_values(ascending=False)

    top1_user = user_mean_exp.index[0]
    top1_user_mean = float(user_mean_exp.iloc[0])

    cutoff_99 = float(user_mean_exp.quantile(0.99))
    top1pct_users = set(user_mean_exp[user_mean_exp >= cutoff_99].index)

    scenarios = [
        ("baseline_all_users", df),
        ("exclude_top1_user", df[df["user_id"] != top1_user].copy()),
        ("exclude_top1pct_users", df[~df["user_id"].isin(top1pct_users)].copy()),
    ]

    meta = {
        "top1_user": top1_user,
        "top1_user_mean_expense_30d": top1_user_mean,
        "top1pct_cutoff_mean_expense_30d": cutoff_99,
        "top1pct_user_count": len(top1pct_users),
    }
    return scenarios, user_mean_exp, meta


def plot_spend_contribution(user_mean_exp: pd.Series, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    top = user_mean_exp.head(20)
    plt.figure()
    plt.bar(top.index.astype(str), top.values)
    plt.title("Top 20 Users by Mean 30d Expense")
    plt.xlabel("user_id")
    plt.ylabel("mean expense_total_30d")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "sensitivity_top20_users_mean_expense.png", dpi=150)
    plt.close()

    # Contribution curve (Pareto-ish)
    cum_share = (user_mean_exp / user_mean_exp.sum()).cumsum()
    plt.figure()
    plt.plot(np.arange(1, len(cum_share) + 1), cum_share.values)
    plt.title("Cumulative Share of Mean 30d Expense (Users Sorted High→Low)")
    plt.xlabel("Top N users")
    plt.ylabel("Cumulative share")
    plt.tight_layout()
    plt.savefig(fig_dir / "sensitivity_cumulative_expense_share.png", dpi=150)
    plt.close()


def run_one_scenario(name: str, df: pd.DataFrame):
    # Split first (strict time-based)
    train_df, val_df, test_df = time_split_3way(df, train_frac=0.7, val_frac=0.15)

    # Labels from TRAIN ONLY
    label_thr = build_labels_train_only(train_df, val_df, test_df, q=0.80)

    # Features
    X_train = train_df[FEATURES_NUM + FEATURES_CAT]
    y_train = train_df[Y_COL].astype(int)

    X_val = val_df[FEATURES_NUM + FEATURES_CAT]
    y_val = val_df[Y_COL].astype(int)

    X_test = test_df[FEATURES_NUM + FEATURES_CAT]
    y_test = test_df[Y_COL].astype(int)

    # Imbalance weight
    n_pos = int(y_train.sum())
    n_neg = int((1 - y_train).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1))

    # Preprocess (tree)
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    # Train XGB + choose n_estimators on VAL PR-AUC, choose threshold on VAL F1
    model, best_n, best_val_pr, proba_thr, best_val_f1 = train_xgb_manual_tuning(
        X_train_t, y_train, X_val_t, y_val, scale_pos_weight
    )

    # Evaluate on TEST
    test_proba = model.predict_proba(X_test_t)[:, 1]
    test_metrics = evaluate(y_test, test_proba, proba_thr)

    return {
        "scenario": name,
        "rows_total": int(len(df)),
        "users_total": int(df["user_id"].nunique()),
        "label_threshold_train_q80": float(label_thr),
        "train_pos_rate": float(y_train.mean()),
        "val_pos_rate": float(y_val.mean()),
        "test_pos_rate": float(y_test.mean()),
        "xgb_best_n_estimators": int(best_n),
        "val_pr_auc_best_n": float(best_val_pr),
        "val_f1_at_best_thr": float(best_val_f1),
        "proba_threshold_val_f1": float(proba_thr),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "test_pr_auc": float(test_metrics["pr_auc"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_cm": str(test_metrics["cm"].tolist()),
    }


def main():
    p = Paths()
    p.report_dir.mkdir(parents=True, exist_ok=True)
    p.fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(p.data_path)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    # basic checks
    for c in [RATIO_COL, "expense_total_30d"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    scenarios, user_mean_exp, meta = scenario_filters(df)

    print("Outlier meta:")
    for k, v in meta.items():
        print(f" - {k}: {v}")

    plot_spend_contribution(user_mean_exp, p.fig_dir)

    results = []
    for name, sdf in scenarios:
        print(f"\nRunning scenario: {name} | rows={len(sdf)} | users={sdf['user_id'].nunique()}")
        results.append(run_one_scenario(name, sdf))

    out = pd.DataFrame(results).sort_values("val_f1_at_best_thr", ascending=False)
    out_path = p.report_dir / "sensitivity_outliers.csv"
    out.to_csv(out_path, index=False)

    print("\n✅ Wrote:", out_path)
    print(out[[
        "scenario",
        "rows_total",
        "users_total",
        "xgb_best_n_estimators",
        "val_f1_at_best_thr",
        "test_pr_auc",
        "test_roc_auc",
        "test_accuracy",
    ]])


if __name__ == "__main__":
    main()