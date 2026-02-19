from pathlib import Path
import numpy as np
import pandas as pd


def build_daily_user(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user per day."""
    # Create income/expense columns
    df = df.copy()
    df["income_amt"] = np.where(df["direction"] == "Income", df["amount"], 0.0)
    df["expense_amt"] = np.where(df["direction"] == "Expense", df["amount"], 0.0)
    df["net"] = df["income_amt"] - df["expense_amt"]

    daily_user = (
        df.groupby(["date", "user_id"], as_index=False)
        .agg(
            income_total=("income_amt", "sum"),
            expense_total=("expense_amt", "sum"),
            net=("net", "sum"),
            txn_count=("txn_uid", "count"),
        )
        .sort_values(["date", "user_id"])
    )
    return daily_user


def build_monthly_category(df: pd.DataFrame) -> pd.DataFrame:
    """One row per month per category (expenses only)."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    expenses = df[df["direction"] == "Expense"].copy()

    monthly_cat = (
        expenses.groupby(["month", "category"], as_index=False)
        .agg(
            total_expense=("amount", "sum"),
            txn_count=("txn_uid", "count"),
            unique_users=("user_id", "nunique"),
        )
        .sort_values(["month", "total_expense"], ascending=[True, False])
    )
    return monthly_cat

def build_user_30d_features(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """One row per user with rolling-window features (last N days)."""

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    window_end = df["date"].max()
    window_start = window_end - pd.Timedelta(days=window_days - 1)

    w = df[(df["date"] >= window_start) & (df["date"] <= window_end)].copy()

    # basic columns
    w["income_amt"] = np.where(w["direction"] == "Income", w["amount"], 0.0)
    w["expense_amt"] = np.where(w["direction"] == "Expense", w["amount"], 0.0)

    # txn count + active days
    base = (
        w.groupby("user_id", as_index=False)
        .agg(
            txn_count_30d=("txn_uid", "count"),
            active_days_30d=("date", lambda s: s.dt.date.nunique()),
            income_total_30d=("income_amt", "sum"),
            expense_total_30d=("expense_amt", "sum"),
            payment_mode_diversity_30d=("payment_mode", lambda s: s.nunique(dropna=True)),
        )
    )

    base["net_total_30d"] = base["income_total_30d"] - base["expense_total_30d"]
    base["days_in_window"] = window_days
    base["window_start"] = window_start.date()
    base["window_end"] = window_end.date()

    # expense-only stats
    exp = w[w["direction"] == "Expense"].copy()

    exp_stats = (
        exp.groupby("user_id", as_index=False)
        .agg(
            median_expense_txn_30d=("amount", "median"),
            expense_std_30d=("amount", "std"),
            category_diversity_30d=("category", lambda s: s.nunique(dropna=True)),
        )
    )

    # top category + share
    if len(exp) > 0:
        cat_sum = (
            exp.groupby(["user_id", "category"], as_index=False)["amount"].sum()
            .rename(columns={"amount": "cat_expense"})
        )

        # pick top category per user
        cat_sum = cat_sum.sort_values(["user_id", "cat_expense"], ascending=[True, False])
        top_cat = cat_sum.groupby("user_id").head(1).copy()
        top_cat = top_cat.rename(columns={"category": "top_category_30d"})

        # compute share of top category
        total_exp = exp.groupby("user_id", as_index=False)["amount"].sum().rename(columns={"amount": "total_expense_user"})
        top_cat = top_cat.merge(total_exp, on="user_id", how="left")
        top_cat["top_category_share_30d"] = top_cat["cat_expense"] / top_cat["total_expense_user"]
        top_cat = top_cat[["user_id", "top_category_30d", "top_category_share_30d"]]
    else:
        top_cat = pd.DataFrame(columns=["user_id", "top_category_30d", "top_category_share_30d"])

    # merge all
    out = base.merge(exp_stats, on="user_id", how="left").merge(top_cat, on="user_id", how="left")

    # derived averages (avoid divide by zero)
    out["avg_expense_per_day_30d"] = out["expense_total_30d"] / out["days_in_window"]
    out["avg_txn_per_active_day_30d"] = out["txn_count_30d"] / out["active_days_30d"].replace(0, np.nan)

    # fill some nulls where user had no expenses in window
    out["median_expense_txn_30d"] = out["median_expense_txn_30d"].fillna(0.0)
    out["expense_std_30d"] = out["expense_std_30d"].fillna(0.0)
    out["category_diversity_30d"] = out["category_diversity_30d"].fillna(0).astype(int)
    out["top_category_share_30d"] = out["top_category_share_30d"].fillna(0.0)
    out["top_category_30d"] = out["top_category_30d"].fillna("None")

    # sort for readability
    out = out.sort_values("expense_total_30d", ascending=False)

    return out


def main():
    processed_dir = Path("data/processed")
    in_path = processed_dir / "transactions_clean.parquet"

    # 1) Validate input exists
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing {in_path}. Generate it first from your EDA/cleaning notebook."
        )

    # 2) Load cleaned canonical dataset
    df = pd.read_parquet(in_path)

    # 3) Basic sanity checks (company-style)
    required = {"txn_uid", "user_id", "date", "direction", "category", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"transactions_clean.parquet is missing required columns: {missing}")

    # 4) Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_date = df["date"].isna().mean()
    if bad_date > 0.0:
        # Keep pipeline running but warn; you can later turn this into a hard failure
        print(f"WARNING: {bad_date:.2%} rows have invalid dates even after cleaning.")

    # 5) Build marts
    daily_user = build_daily_user(df)
    monthly_cat = build_monthly_category(df)
    user_30d = build_user_30d_features(df, window_days=30)


    # 6) Write outputs
    processed_dir.mkdir(parents=True, exist_ok=True)
    daily_user.to_csv(processed_dir / "mart_daily_user.csv", index=False)
    monthly_cat.to_csv(processed_dir / "mart_monthly_category.csv", index=False)
    user_30d.to_csv(processed_dir / "mart_user_30d_features.csv", index=False)


    # 7) Print quick summary
    print("✅ Wrote marts:")
    print("-", processed_dir / "mart_daily_user.csv", "| rows:", len(daily_user))
    print("-", processed_dir / "mart_monthly_category.csv", "| rows:", len(monthly_cat))
    print("-", processed_dir / "mart_user_30d_features.csv", "| rows:", len(user_30d))
    

if __name__ == "__main__":
    main()