# SpendSense — EDA & Data Quality Summary (BudgetWise)

## Goal
Build a company-standard analytics foundation from messy transaction data:
- quantify data quality issues
- create a canonical cleaned dataset (`tx_clean`)
- generate initial EDA insights + reusable artifacts for reporting / BI

---

## Data Sources
- **Dirty source (primary):** `budgetwise_synthetic_dirty.csv`
- **Clean reference (baseline sanity checks):** `budgetwise_finance_dataset.csv`

Schema (both):  
`transaction_id, user_id, date, transaction_type, category, amount, payment_mode, location, notes`

---

## Data Quality Checks (Evidence)
### Clean vs Dirty (post-clean) comparison
| dataset | rows | date_fail_rate | amount_fail_rate | payment_mode_unique | category_unique |
|---|---:|---:|---:|---:|---:|
| CLEAN | 15,900 | 0.030566 | 0.018679 | 8 | 31 |
| DIRTY (post-clean) | 15,032 | 0.021754 | 0.011309 | 61 | 211 |

**Interpretation**
- The dirty dataset simulates real production issues: inconsistent category labels, inconsistent payment modes, mixed date formats, currency formatting, and broken identifiers.
- The clean dataset is used as a reference baseline for expected cardinality (categories/payment modes) and for sanity-checking distributions.

---

## Key Findings from Dirty Source
### 1) Exact duplicate rows
- Dirty rows: **15,836**
- After dropping exact duplicates: **15,032**
- Removed: **804** exact duplicate rows

### 2) Date parsing issues (mixed formats)
Observed formats include:
- ISO: `YYYY-MM-DD`
- short dash dates: `DD-MM-YY`
- slash dates: `MM/DD/YYYY` and `DD/MM/YY`
- month-name formats: `October 18 2022`

Approach:
- used strict multi-format parsing to reduce failures deterministically

Result:
- Post-clean date parse failure rate: **~2.18%**

### 3) Amount formatting issues
Examples:
- currency symbols: `$127`, `₹54,120`
- thousands separators: `3,352`

Approach:
- normalized amount strings to numeric floats (strip currency + commas)

Result:
- Post-clean amount parse failure rate: **~1.13%**

### 4) Broken transaction identifiers
- Same `transaction_id` appears with conflicting values (different users/dates/amounts/categories).
- Conclusion: `transaction_id` is **not a reliable primary key**.

Approach:
- created canonical `txn_uid` fingerprint from stable fields:
  `(user_id, date_parsed, direction, category, amount, payment_mode, location)`

Result:
- `txn_uid` duplicates were minimal (**18**) and represent near-identical transactions.

---

## Canonical Dataset Output
**Saved as:** `data/processed/transactions_clean.parquet`

Canonical columns:
- `txn_uid` (primary key)
- `source_transaction_id`
- `user_id`
- `date`
- `direction` (Income/Expense)
- `category`
- `amount` (numeric)
- `payment_mode`
- `location`
- `notes`

Minimum validity rule:
- drop rows missing any of: `date`, `amount`, `direction`

---

## EDA Artifacts Generated
Saved under `reports/figures/`:
- `daily_net_cashflow_all_users.png`  
  Shows net inflow/outflow trend over time across all users.
- `top10_expense_categories.png`  
  Highlights dominant spending categories by total spend.

---

## Next Steps
1) Build **BI marts** (Power BI-ready CSVs) from `transactions_clean.parquet`
2) Implement **SpendSense Insights Engine**
   - weekly/monthly spend trend deltas
   - category mix shifts
   - anomaly detection for unusual transactions
3) Add **baseline forecasting**
   - short-horizon expense forecast per user/category