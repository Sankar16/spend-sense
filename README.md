# SpendSense — Predicting Spend Spikes + Measuring Impact (A/B + Causal Inference)

## TL;DR
**SpendSense** is an end-to-end Product Data Science project that predicts **30-day spend spikes** for users and proposes how to reduce them using **alerts + budget recommendations**.  
It includes a production-style pipeline (raw → canonical → marts), dashboard previews, a time-aware ML model (XGBoost), robustness checks (outliers), and an experimentation + causal inference plan (A/B, DiD, IPW).

**Key results**
- **Spike prediction model (XGBoost):** ROC-AUC ≈ **0.864**, PR-AUC ≈ **0.559** (time-based split)
- **Outlier robustness:** Removing the top spender / top 1% spenders keeps metrics stable (PR-AUC stays ≈ **0.559–0.563**)
- **Causal demo (simulated intervention):** A/B + DiD show a **~$190 reduction** in next-30-day spend (CI fully negative) under a 12% treatment effect assumption.

---

## 1) Business Problem (Product DS framing)
**Stakeholders**
- **Product:** wants to reduce negative customer experiences from unexpected spending
- **Finance:** wants fewer overdraft/late fee events (proxy: spend spikes)
- **Customer Success:** wants actionable insights (top categories driving increases)

**Goal**
Detect users likely to have a **spend spike in the next 30 days** and trigger a lightweight intervention:
- “Spend alert” when user is at risk
- Personalized budget recommendations (top categories, essentials share, trend changes)

**Success metrics**
- Primary: **Spend spike rate** reduction (binary)
- Secondary: **Next-30-day spend** reduction (continuous)
- Guardrails (planned): opt-out rate, alert fatigue, retention/churn proxies

---

## 2) Data
**Source**
A personal finance transactions dataset containing:
- `user_id`, `date`, `direction` (Income/Expense), `category`, `amount`, `payment_mode`, `location`, `notes`

**Core challenges addressed**
- messy dates (many formats), currency symbols, commas
- inconsistent text categories (typos like “Foood”, “Saalry”, etc.)
- duplicates / conflicting transaction IDs
- missing values (notes/payment_mode/location)

---

## 3) Pipeline Architecture (company-style)
### Raw → Canonical → Marts
- **Raw:** `data/raw/archive/*.csv`
- **Canonical:** `data/processed/transactions_clean.parquet`
  - standardized schema
  - cleaned dates and numeric amounts
  - normalized categories/payment modes
  - deduped transactions (stable `txn_uid`)
- **Marts (analytics ready)**
  1. `mart_daily_user.csv`  
     Daily income/expense/net by user
  2. `mart_monthly_category.csv`  
     Monthly totals by category + user counts
  3. `mart_user_30d_features.csv`  
     Snapshot features for last 30d (activity, diversity, spend stats)
  4. `mart_user_30d_trends.csv`  
     Trend features (last7 vs prev7, essentials share, large txn count)

**Why marts?**
Marts align data to “business questions” and make dashboarding/ML repeatable:
- Product: daily net cashflow + top categories
- DS: consistent features and labels for modeling
- BI: stable aggregations for dashboards

---

## 4) Exploratory Analysis + Dashboard Previews
This project includes dashboard preview scripts that generate images (saved to `reports/figures/`) for:
- Net cashflow trend over time
- Top expense categories (latest month)
- Monthly category spend (stacked bar)
- User distribution + top spenders

These previews are intended as a lightweight alternative to PowerBI while keeping the same metrics.

---

## 5) ML Problem Definition
### Prediction Target: “Spend spike next 30 days”
Instead of predicting absolute spend (unstable), I predict a **spike event** to support alerting:

- Compute `spike_ratio_next_30d = (next_30d_expense_total + 1) / (expense_total_30d + 1)`
- Label as spike if ratio ≥ **train-only 80th percentile** (prevents leakage)

**Why this target works**
- gives a clear binary decision for alerting
- supports threshold tuning based on cost/benefit (precision/recall)
- aligns with product intervention (reduce spikes)

---

## 6) Model Training + Evaluation (time-aware)
### Train/Val/Test split (strict time ordering)
To avoid leakage, I split by `as_of_date`:
- Train: earliest 70%
- Val: next 15% (threshold tuning)
- Test: most recent 15%

### Models
- Logistic Regression (interpretable baseline)
- Random Forest
- **XGBoost (best)**

### XGBoost results (time split)
- **TEST ROC-AUC:** ~**0.864**
- **TEST PR-AUC:** ~**0.559**
- Threshold chosen on **VAL F1** and applied to TEST

> Why PR-AUC matters here: spike events are ~20% of cases, so PR-AUC is more informative than accuracy.

---

## 7) Outlier Robustness (Senior DS check)
Spend data is heavy-tailed. I tested whether performance depends on a few power users:

Scenarios
1) Baseline (all users)  
2) Exclude top spender user  
3) Exclude top 1% spenders (by mean 30-day expense)

Result: metrics are stable:
- TEST PR-AUC stays ~**0.559 → 0.563**
- ROC-AUC stable ~**0.864–0.865**

**Conclusion:** the model generalizes and is not driven by a single extreme account.

---

## 8) Product Decisioning (how this would ship)
**Who gets an alert?**
- Users with `risk_proba >= threshold` (from validation tuning)

**How often?**
- Weekly scoring window (aligned with feature generation frequency)
- Alert suppression rules (planned):
  - don’t alert if user recently alerted in last X days
  - don’t alert if predicted risk barely above threshold (hysteresis)
  - cap alerts per month

**What the alert contains**
- top category driver last 30d
- growth trend (last7 vs prev7)
- essentials share (budget stability)

---

## 9) Impact Measurement (A/B + Causal Inference)
### A/B test design
Population: high-risk users (eligible cohort)  
Randomization: blocked by `as_of_date`  
Outcome:
- spike rate reduction (binary)
- next-30-day spend reduction (continuous)

### Offline causal demo (methodology)
Because we don’t have real treatment logs, I simulate an intervention:
- Treated users have **12% lower** next-30-day spend (ground truth)

Then I estimate impact using:
1) **A/B difference in means** (ATE)
2) **Difference-in-Differences (DiD)** on spend
3) **Propensity weighting (IPW)** for observational settings

**Example output (one run)**
- Spike spend threshold (q80): **1653.6**
- A/B spike rate ATE: **-3.11pp** (CI crosses 0; limited sample)
- A/B spend ATE: **-193.8** (CI fully negative)
- DiD spend: **-192.3** (CI fully negative)
- Observational naive vs IPW: both directionally consistent here

**Why this matters**
This shows readiness to measure real product impact (not just model metrics), and demonstrates causal tooling needed for Product DS roles.

---

## 10) Limitations + Next Steps
- Use real treatment logs (alerts sent, viewed, ignored)
- Add alert fatigue / churn guardrails
- Calibrate probabilities (Platt/Isotonic)
- Drift monitoring: category distribution, spend distribution, score stability
- Power analysis for A/B: estimate required sample size for spike-rate significance
- Consider uplift modeling (who benefits most from alert)

---

## Repro Steps
Create env + run key stages:

```bash
# Build marts
python -m src.spendsense.data.build_marts
python -m src.spendsense.data.build_mart_user_30d_trends

# Build training dataset
python -m src.spendsense.modeling.build_training_dataset

# Train models (time split + tuning)
python -m src.spendsense.modeling.train_baseline

# Outlier sensitivity
python -m src.spendsense.modeling.sensitivity_outliers

# Causal / A/B demo (simulated treatment effect)
python -m src.spendsense.modeling.causal_ab_test_demo
```

## What to look at
- `reports/baseline_metrics.csv` — model metrics (LogReg/RF/XGB)
- `reports/sensitivity_outliers.csv` — robustness check
- `reports/causal_ab_results.csv` — causal + A/B impact estimates
