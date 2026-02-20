# src/spendsense/modeling/causal_ab_test_demo.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


@dataclass(frozen=True)
class Paths:
    data_path: Path = Path("data/processed/training_dataset.parquet")
    metrics_path: Path = Path("reports/baseline_metrics.csv")
    xgb_bundle_path: Path = Path("reports/models/baseline_xgb.joblib")
    out_csv: Path = Path("reports/causal_ab_results.csv")


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

EXPENSE_PRE = "expense_total_30d"
EXPENSE_POST = "next_30d_expense_total"


def read_xgb_threshold(metrics_path: Path) -> float:
    m = pd.read_csv(metrics_path)
    row = m.loc[m["model"] == "XGBoost"]
    if row.empty:
        raise ValueError("XGBoost row not found in reports/baseline_metrics.csv")
    return float(row["threshold"].iloc[0])


def bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray, fn, n_boot=800, seed=42):
    rng = np.random.default_rng(seed)
    n_a = len(values_a)
    n_b = len(values_b)
    stats = []
    for _ in range(n_boot):
        sa = values_a[rng.integers(0, n_a, size=n_a)]
        sb = values_b[rng.integers(0, n_b, size=n_b)]
        stats.append(fn(sa) - fn(sb))
    stats = np.array(stats)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    p = Paths()
    df = pd.read_parquet(p.data_path)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])

    # Load XGB bundle (preprocess + model)
    bundle = joblib.load(p.xgb_bundle_path)
    pre = bundle["preprocess"]
    model = bundle["model"]

    # Use model threshold to define "eligible" high-risk cohort
    risk_thr = read_xgb_threshold(p.metrics_path)

    # Score everyone
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    X_t = pre.transform(X)
    df["risk_proba"] = model.predict_proba(X_t)[:, 1]

    # Define experiment population: high-risk users
    df["eligible"] = (df["risk_proba"] >= risk_thr).astype(int)
    eligible = df[df["eligible"] == 1].copy()

    # Use a time slice for "experiment period" (e.g., last 15% dates = test-like period)
    dates = sorted(eligible["as_of_date"].unique())
    cut = int(len(dates) * 0.85)
    exp_dates = set(dates[cut:])  # latest 15%
    exp = eligible[eligible["as_of_date"].isin(exp_dates)].copy()

    # -----------------------------
    # A/B TEST (blocked randomization by date)
    # -----------------------------
    rng = np.random.default_rng(7)
    exp["treatment"] = 0

    for _, g in exp.groupby("as_of_date"):
        idx = g.index.to_numpy(copy=True)  # writable
        rng.shuffle(idx)
        half = len(idx) // 2
        exp.loc[idx[:half], "treatment"] = 1  # treatment group

    # -----------------------------
    # SIMULATE true causal effect (because we don't have real intervention logs)
    # Example: alert reduces next 30-day spend by 12% among treated
    # -----------------------------
    TRUE_SPEND_REDUCTION = 0.12  # 12% reduction
    exp["next_30d_spend_cf"] = exp[EXPENSE_POST].astype(float)

    # apply effect to treated
    exp.loc[exp["treatment"] == 1, "next_30d_spend_cf"] = (
        exp.loc[exp["treatment"] == 1, "next_30d_spend_cf"] * (1.0 - TRUE_SPEND_REDUCTION)
    )

    # -----------------------------
    # Define spike outcome using ABSOLUTE spend (stable vs ratios when baseline≈0)
    # Use q80 of counterfactual spend within experiment cohort
    # -----------------------------
    spike_spend_thr = float(exp["next_30d_spend_cf"].quantile(0.80))
    exp["spike_cf"] = (exp["next_30d_spend_cf"] >= spike_spend_thr).astype(int)

    # -----------------------------
    # Estimator 1: A/B difference in spike rate (ATE on binary)
    # -----------------------------
    t = exp[exp["treatment"] == 1]
    c = exp[exp["treatment"] == 0]

    spike_rate_t = float(t["spike_cf"].mean())
    spike_rate_c = float(c["spike_cf"].mean())
    ate_spike = spike_rate_t - spike_rate_c
    ci_spike = bootstrap_ci(t["spike_cf"].to_numpy(), c["spike_cf"].to_numpy(), fn=np.mean)

    # -----------------------------
    # Estimator 2: A/B difference in spend (ATE on continuous)
    # -----------------------------
    spend_t = float(t["next_30d_spend_cf"].mean())
    spend_c = float(c["next_30d_spend_cf"].mean())
    ate_spend = spend_t - spend_c
    ci_spend = bootstrap_ci(
        t["next_30d_spend_cf"].to_numpy(), c["next_30d_spend_cf"].to_numpy(), fn=np.mean
    )

    # -----------------------------
    # Estimator 3: DiD on spend using pre/post (delta = post - pre)
    # -----------------------------
    t_delta = (t["next_30d_spend_cf"] - t[EXPENSE_PRE]).to_numpy()
    c_delta = (c["next_30d_spend_cf"] - c[EXPENSE_PRE]).to_numpy()
    did = float(t_delta.mean() - c_delta.mean())
    ci_did = bootstrap_ci(t_delta, c_delta, fn=np.mean)

    # -----------------------------
    # Observational demo: biased treatment assignment + IPW correction
    # -----------------------------
    obs = exp.copy()

    # biased assignment: higher risk_proba => more likely treated (creates confounding)
    p_treat = np.clip(0.05 + 0.9 * obs["risk_proba"], 0.05, 0.95)
    obs["treatment_obs"] = (rng.random(len(obs)) < p_treat).astype(int)

    # simulate same true effect on treated_obs
    obs["next_30d_spend_obs"] = obs[EXPENSE_POST].astype(float)
    obs.loc[obs["treatment_obs"] == 1, "next_30d_spend_obs"] *= (1.0 - TRUE_SPEND_REDUCTION)

    # define spike using SAME spend threshold as RCT definition
    obs["spike_obs"] = (obs["next_30d_spend_obs"] >= spike_spend_thr).astype(int)

    # Naive difference (biased)
    naive_spike = float(
        obs.loc[obs["treatment_obs"] == 1, "spike_obs"].mean()
        - obs.loc[obs["treatment_obs"] == 0, "spike_obs"].mean()
    )

    # IPW: estimate propensity via logistic regression using a few covariates
    covars = ["risk_proba", EXPENSE_PRE, "expense_growth_ratio_7d", "essentials_share_30d", "large_txn_cnt_30d"]
    Xp = obs[covars].fillna(0.0).to_numpy()
    yp = obs["treatment_obs"].to_numpy()

    prop = LogisticRegression(max_iter=2000)
    prop.fit(Xp, yp)
    ps = np.clip(prop.predict_proba(Xp)[:, 1], 0.01, 0.99)
    obs["ps"] = ps

    # Stabilized weights
    pA = float(obs["treatment_obs"].mean())
    obs["w"] = np.where(obs["treatment_obs"] == 1, pA / obs["ps"], (1 - pA) / (1 - obs["ps"]))

    # IPW ATE on spike
    wt_t = obs.loc[obs["treatment_obs"] == 1, "w"]
    wt_c = obs.loc[obs["treatment_obs"] == 0, "w"]
    ipw_spike = float(
        (obs.loc[obs["treatment_obs"] == 1, "spike_obs"] * wt_t).sum() / wt_t.sum()
        - (obs.loc[obs["treatment_obs"] == 0, "spike_obs"] * wt_c).sum() / wt_c.sum()
    )

    # Diagnostics
    prop_loss = float(log_loss(yp, ps))

    results = pd.DataFrame(
        [
            {
                "cohort_rows": int(len(exp)),
                "unique_users": int(exp["user_id"].nunique()),
                "threshold_risk_proba": float(risk_thr),
                "spike_spend_threshold_demo": float(spike_spend_thr),
                "true_spend_reduction": float(TRUE_SPEND_REDUCTION),
                # A/B
                "ab_spike_rate_treat": spike_rate_t,
                "ab_spike_rate_ctrl": spike_rate_c,
                "ab_ate_spike": ate_spike,
                "ab_ate_spike_ci_low": ci_spike[0],
                "ab_ate_spike_ci_high": ci_spike[1],
                "ab_spend_treat": spend_t,
                "ab_spend_ctrl": spend_c,
                "ab_ate_spend": ate_spend,
                "ab_ate_spend_ci_low": ci_spend[0],
                "ab_ate_spend_ci_high": ci_spend[1],
                "did_spend": did,
                "did_spend_ci_low": ci_did[0],
                "did_spend_ci_high": ci_did[1],
                # Observational demo
                "obs_naive_ate_spike": naive_spike,
                "obs_ipw_ate_spike": ipw_spike,
                "propensity_logloss": prop_loss,
            }
        ]
    )

    p.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(p.out_csv, index=False)

    print("✅ Wrote:", p.out_csv)
    print(results.T)


if __name__ == "__main__":
    main()