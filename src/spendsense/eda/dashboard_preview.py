from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    processed = Path("data/processed")
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(processed / "mart_daily_user.csv", parse_dates=["date"])
    monthly = pd.read_csv(processed / "mart_monthly_category.csv", parse_dates=["month"])
    users = pd.read_csv(processed / "mart_user_30d_features.csv")

    # -------- Page 1 preview: Net cashflow over time (all users) --------
    daily_all = daily.groupby("date", as_index=False)[["income_total", "expense_total", "net", "txn_count"]].sum()

    plt.figure()
    plt.plot(daily_all["date"], daily_all["net"])
    plt.title("Daily Net Cashflow (All Users)")
    plt.xlabel("Date")
    plt.ylabel("Net")
    plt.tight_layout()
    plt.savefig(fig_dir / "dashboard_page1_net_cashflow.png", dpi=150)
    plt.close()

    # -------- Page 1 preview: Top categories (latest month) --------
    latest_month = monthly["month"].max()
    m_latest = monthly[monthly["month"] == latest_month].copy()
    top10 = m_latest.sort_values("total_expense", ascending=False).head(10)

    plt.figure()
    plt.bar(top10["category"], top10["total_expense"])
    plt.title(f"Top 10 Expense Categories ({latest_month.date()})")
    plt.xlabel("Category")
    plt.ylabel("Total Expense")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "dashboard_page1_top_categories.png", dpi=150)
    plt.close()

    # -------- Page 2 preview: Monthly spend (top 8 categories) --------
    top_cats = (
        monthly.groupby("category", as_index=False)["total_expense"].sum()
        .sort_values("total_expense", ascending=False)
        .head(8)["category"]
        .tolist()
    )
    m2 = monthly[monthly["category"].isin(top_cats)].copy()

    pivot = (
        m2.pivot_table(
            index="month",
            columns="category",
            values="total_expense",
            aggfunc="sum",
        )
        .fillna(0)
        .sort_index()
    )

    # Convert index to friendly labels (avoids PeriodConverter/freq issues)
    pivot.index = pivot.index.strftime("%Y-%m")

    ax = pivot.plot(kind="bar", stacked=True)
    ax.set_title("Monthly Expense by Category (Top 8)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Expense")
    plt.tight_layout()
    plt.savefig(fig_dir / "dashboard_page2_monthly_category_stacked.png", dpi=150)
    plt.close()
    

    # -------- Page 3 preview: User scatter (30d) --------
    plt.figure()
    plt.scatter(users["expense_total_30d"], users["expense_std_30d"])
    plt.title("User Behavior (30d): Expense vs Volatility")
    plt.xlabel("Expense Total (30d)")
    plt.ylabel("Expense Std Dev (30d)")
    plt.tight_layout()
    plt.savefig(fig_dir / "dashboard_page3_user_scatter.png", dpi=150)
    plt.close()

    # -------- Page 3 preview: Top 10 users by expense --------
    top_users = users.sort_values("expense_total_30d", ascending=False).head(10)

    plt.figure()
    plt.bar(top_users["user_id"], top_users["expense_total_30d"])
    plt.title("Top 10 Users by Expense (Last 30 Days)")
    plt.xlabel("User")
    plt.ylabel("Expense Total (30d)")
    plt.tight_layout()
    plt.savefig(fig_dir / "dashboard_page3_top_users.png", dpi=150)
    plt.close()

    print("✅ Dashboard preview images saved to reports/figures/")


if __name__ == "__main__":
    main()