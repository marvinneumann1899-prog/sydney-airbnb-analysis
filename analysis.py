"""Airbnb in Sydney — exploratory analysis for the city planners (CRISP-DM phases 2-3).

Data: Inside Airbnb, Sydney snapshot 2026-06-16 (listings.csv.gz, reviews.csv.gz).
Run:  .venv/bin/python analysis.py
Writes tables to results/ and figures to figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
RES = HERE / "results"
FIG = HERE / "figures"
RES.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SNAPSHOT = pd.Timestamp("2026-06-16")
LONG_TERM_NIGHTS = 30          # brief: short-term < 30 nights, long-term >= 30
MULTI_HOST = 2                 # host with 2+ listings
COMMERCIAL_HOST = 10           # host with 10+ listings

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
C_TOP, C_REST = "#1f4e79", "#a6a6a6"


# --- Data preparation ---------------------------------------------------------
def load_listings() -> pd.DataFrame:
    cols = ["id", "neighbourhood_cleansed", "room_type", "property_type", "accommodates",
            "host_id", "calculated_host_listings_count", "minimum_nights", "price",
            "number_of_reviews", "first_review", "last_review", "availability_365",
            "number_of_reviews_ltm"]
    df = pd.read_csv(RAW / "listings.csv.gz", usecols=cols, low_memory=False)
    df["price"] = pd.to_numeric(df["price"].astype(str).str.replace(r"[$,]", "", regex=True),
                                errors="coerce")
    df["first_review"] = pd.to_datetime(df["first_review"])
    df["last_review"] = pd.to_datetime(df["last_review"])
    df["term"] = np.where(df["minimum_nights"] >= LONG_TERM_NIGHTS, "long-term (30+ nights)",
                          "short-term (<30 nights)")
    df["host_type"] = pd.cut(df["calculated_host_listings_count"], [0, 1, COMMERCIAL_HOST - 1, np.inf],
                             labels=["single-listing host", "2-9 listings", "10+ listings"])
    return df


def add_area(df: pd.DataFrame, top10: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["area"] = np.where(df["neighbourhood_cleansed"].isin(top10), "Top 10", "Rest")
    return df


def share(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Row-normalised shares (%) of `col` for Top 10 vs Rest."""
    t = pd.crosstab(df["area"], df[col], normalize="index").mul(100).round(1)
    return t.loc[["Top 10", "Rest"]]


# --- Question 0: top 10 neighbourhoods ---------------------------------------
def q0_top10(df: pd.DataFrame) -> list[str]:
    vc = df["neighbourhood_cleansed"].value_counts()
    t = pd.DataFrame({"listings": vc, "share_%": (vc / vc.sum() * 100).round(1)})
    t["cumulative_%"] = t["share_%"].cumsum().round(1)
    t.head(10).to_csv(RES / "q0_top10_neighbourhoods.csv")
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    top = t.head(10).iloc[::-1]
    ax.barh(top.index, top["listings"], color=C_TOP)
    for y, (n, s) in enumerate(zip(top["listings"], top["share_%"])):
        ax.text(n + 40, y, f"{n:,}  ({s:.1f} %)", va="center", fontsize=8)
    ax.set_xlabel("Listings")
    ax.set_title(f"Top 10 of {len(vc)} LGAs hold {t['share_%'].head(10).sum():.1f} % of all "
                 f"{vc.sum():,} listings", loc="left", fontsize=9)
    ax.set_xlim(0, top["listings"].max() * 1.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_top10.png")
    plt.close(fig)
    return list(t.head(10).index)


# --- Questions 1-4: Top 10 vs Rest -------------------------------------------
def q1_to_q4(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    out["q1_room_type"] = share(df, "room_type")
    top_props = df["property_type"].value_counts().head(6).index
    df_p = df.assign(property_type=df["property_type"].where(df["property_type"].isin(top_props), "Other"))
    out["q1_property_type"] = share(df_p, "property_type")
    out["q2_accommodates"] = df.groupby("area")["accommodates"].describe().loc[["Top 10", "Rest"]].round(2)
    bins = pd.cut(df["accommodates"], [0, 2, 4, 6, np.inf], labels=["1-2", "3-4", "5-6", "7+"])
    out["q2_accommodates_bins"] = share(df.assign(acc=bins), "acc")
    out["q3_host_type_share"] = share(df, "host_type")
    hosts = (df.groupby("area")
               .apply(lambda g: pd.Series({
                   "hosts": g["host_id"].nunique(),
                   "listings_from_2plus_hosts": int((g["calculated_host_listings_count"] >= MULTI_HOST).sum()),
                   "listings_from_10plus_hosts": int((g["calculated_host_listings_count"] >= COMMERCIAL_HOST).sum()),
               }), include_groups=False)
               .loc[["Top 10", "Rest"]])
    out["q3_host_counts"] = hosts
    out["q4_term"] = share(df, "term")
    for k, v in out.items():
        v.to_csv(RES / f"{k}.csv")

    # one compact 2x2 figure for Q1-Q4
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2))
    panels = [("q1_room_type", "Q1 Room type (%)"), ("q2_accommodates_bins", "Q2 Guests per listing (%)"),
              ("q3_host_type_share", "Q3 Listings by host portfolio size (%)"),
              ("q4_term", "Q4 Minimum stay (%)")]
    for ax, (key, title) in zip(axes.flat, panels):
        t = out[key]
        cols = [c for c in t.columns if t[c].max() >= 0.5]
        x = np.arange(len(cols))
        ax.bar(x - 0.2, t.loc["Top 10", cols], 0.4, color=C_TOP, label="Top 10")
        ax.bar(x + 0.2, t.loc["Rest", cols], 0.4, color=C_REST, label="Rest")
        ax.set_xticks(x, [str(c).replace(" (", "\n(") for c in cols], fontsize=7)
        ax.set_title(title, loc="left", fontsize=8.5)
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_top10_vs_rest.png")
    plt.close(fig)
    return out


# --- Question 5: time series ---------------------------------------------------
def q5_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Only one public snapshot exists. Proxy: a listing counts as active in month m if it
    received at least one review in the 12 months up to m. Reviews only exist for listings
    still online on 2026-06-16, so the series under-counts listings that left the market
    (survivorship bias) and says nothing about prices in the past."""
    rev = pd.read_csv(RAW / "reviews.csv.gz", usecols=["listing_id", "date"], parse_dates=["date"])
    rev = rev.merge(df[["id", "area"]], left_on="listing_id", right_on="id", how="inner")
    rev["month"] = rev["date"].dt.to_period("M")
    months = pd.period_range("2015-01", "2026-05", freq="M")
    first = rev.groupby(["listing_id", "month"]).size().reset_index()[["listing_id", "month"]]
    first = first.merge(df[["id", "area"]], left_on="listing_id", right_on="id")
    rows = []
    for m in months:
        window = first[(first["month"] > m - 12) & (first["month"] <= m)]
        c = window.groupby("area")["listing_id"].nunique()
        rows.append({"month": m.to_timestamp(), "Top 10": c.get("Top 10", 0), "Rest": c.get("Rest", 0)})
    ts = pd.DataFrame(rows).set_index("month")
    ts.to_csv(RES / "q5_active_listings_proxy.csv")

    # price: cross-section by year of first review (market entry cohort), current prices only
    p = df[df["price"].notna() & df["first_review"].notna()].copy()
    p["entry_year"] = p["first_review"].dt.year
    cohort = (p.groupby(["entry_year", "area"])["price"].median().unstack().loc[2015:2026].round(0))
    cohort.to_csv(RES / "q5_price_by_entry_cohort.csv")

    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    ax.plot(ts.index, ts["Top 10"], color=C_TOP, label="Top 10")
    ax.plot(ts.index, ts["Rest"], color=C_REST, label="Rest")
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-10-31"), color="#f2f2f2", zorder=0)
    ax.text(pd.Timestamp("2020-04-01"), ax.get_ylim()[1] * 0.92, "COVID-19 closures", fontsize=7)
    ax.set_ylabel("Review-active listings\n(trailing 12 months)")
    ax.set_title("Q5 Listings active per month (proxy, survivors of the 2026 snapshot only)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_timeseries_proxy.png")
    plt.close(fig)
    return ts


def main() -> None:
    df = load_listings()
    top10 = q0_top10(df)
    df = add_area(df, top10)
    quality = pd.Series({
        "listings": len(df),
        "price_missing_%": round(df["price"].isna().mean() * 100, 1),
        "price_median_AUD": df["price"].median(),
        "price_p99_AUD": df["price"].quantile(0.99),
        "no_review_yet_%": round(df["first_review"].isna().mean() * 100, 1),
    })
    quality.to_csv(RES / "data_quality.csv")
    price = df.groupby("area")["price"].describe(percentiles=[.25, .5, .75]).loc[["Top 10", "Rest"]].round(0)
    price.to_csv(RES / "price_by_area.csv")
    out = q1_to_q4(df)
    ts = q5_timeseries(df)
    print(quality.to_string(), "\n")
    print("Top 10:", top10, "\n")
    for k, v in out.items():
        print(f"--- {k}\n{v.to_string()}\n")
    print("--- price by area\n", price.to_string())
    print("--- q5 proxy (Jan each year)\n", ts[ts.index.month == 1].to_string())


if __name__ == "__main__":
    main()
