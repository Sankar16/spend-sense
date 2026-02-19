# SpendSense Dashboard Spec (Power BI)

## Data Sources (marts)
- data/processed/mart_daily_user.csv
- data/processed/mart_monthly_category.csv
- data/processed/mart_user_30d_features.csv

## Page 1 — Executive Overview
Purpose: high-level health of spending and cashflow.

Visuals:
1) KPI Cards
- Total Expense
- Total Income
- Net Cashflow
- Active Users
- Transactions

2) Line Chart
- Date vs Net Cashflow
- Optional: 7-day rolling expense

3) Bar Chart (Top Categories)
- Current month top categories by total_expense

Filters:
- Month (Date)
- Optional user_id

## Page 2 — Category Deep Dive
1) Stacked column: monthly spend by category
2) Matrix: category x month (total_expense)
3) Line chart: selected category trend

Filters:
- Month range
- Category

## Page 3 — User Insights (30-day snapshot)
1) Table: Top spenders last 30d
2) Scatter: expense_total_30d vs expense_std_30d (size=txn_count_30d)
3) Bar: top_category_share_30d (top 10 users by expense)

Filters:
- user_id