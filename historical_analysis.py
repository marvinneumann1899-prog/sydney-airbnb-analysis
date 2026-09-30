"""Q5 supplement: published Sydney snapshots, fixed June-2026 top-ten LGAs.

Run after analysis.py. Counts describe each scrape, not occupancy or a census.
Prices are positive advertised nightly AUD; missing prices are never imputed.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW, RES, FIG = ROOT / "data/raw", ROOT / "results", ROOT / "figures"
DATES = ["2025-09-12", "2025-12-12", "2026-03-21", "2026-06-16"]
TOP = set(pd.read_csv(RES / "q0_top10_neighbourhoods.csv").iloc[:, 0])
frames, rows, provenance = {}, [], []
for date in DATES:
    name = "listings.csv.gz" if date == DATES[-1] else f"listings-{date}.csv.gz"
    path = RAW / name
    d = pd.read_csv(path, usecols=["id", "last_scraped", "neighbourhood_cleansed", "price"])
    assert d.id.notna().all() and d.id.is_unique
    assert d.neighbourhood_cleansed.notna().all()
    d["area"] = np.where(d.neighbourhood_cleansed.isin(TOP), "Top 10", "Rest")
    d["price"] = pd.to_numeric(d.price.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    d.loc[~np.isfinite(d.price) | (d.price <= 0), "price"] = np.nan
    frames[date] = d
    for area, g in d.groupby("area"):
        p = g.price.dropna()
        rows.append(dict(date=date, area=area, listings=len(g), priced=len(p),
                         missing_pct=100 * g.price.isna().mean(), mean_aud=p.mean(),
                         median_aud=p.median(), above_10000=int((p > 10000).sum())))
    provenance.append(dict(date=date, file=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           scrape_first=d.last_scraped.min(), scrape_last=d.last_scraped.max(),
                           url=f"https://data.insideairbnb.com/australia/nsw/sydney/{date}/data/listings.csv.gz"))
t = pd.DataFrame(rows).sort_values(["date", "area"])
t.to_csv(RES / "q5_historical_snapshots.csv", index=False, float_format="%.4f")
matched = frames[DATES[-2]].merge(frames[DATES[-1]], on="id", suffixes=("_march", "_june"))
matched = matched.dropna(subset=["price_march", "price_june"])
matched = matched[matched.area_march == matched.area_june]
panel = matched.groupby("area_june").agg(n=("id", "size"), march_mean=("price_march", "mean"),
    june_mean=("price_june", "mean"), march_median=("price_march", "median"), june_median=("price_june", "median"))
panel.to_csv(RES / "q5_matched_prices.csv", float_format="%.4f")
(RES / "q5_history_provenance.json").write_text(json.dumps({"snapshots": provenance,
    "fixed_top10": sorted(TOP), "price_unit": "advertised nightly AUD",
    "note": "2025 prices are missing. Two 2026 price observations; no interpolation or causal claim."}, indent=2))
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6), layout="constrained")
for area, color in [("Top 10", "#1f4e79"), ("Rest", "#777777")]:
    g = t[t.area == area]
    axes[0].plot(pd.to_datetime(g.date), g.listings, marker="o", color=color, label=area)
    g = g[g.priced > 0]
    axes[1].plot(pd.to_datetime(g.date), g.mean_aud, marker="o", color=color, label=area)
for ax in axes:
    ax.set_xticks(pd.to_datetime(DATES), ["Sep 25", "Dec 25", "Mar 26", "Jun 26"])
    ax.tick_params(axis="x", labelsize=7)
axes[0].set(title="Observed listings", ylabel="Listings per scrape")
axes[1].set(title="Mean advertised nightly price", ylabel="AUD, available positive prices")
axes[1].text(.02, .04, "2025: no price observations", transform=axes[1].transAxes, fontsize=7)
axes[0].legend(frameon=False, fontsize=7)
fig.savefig(FIG / "fig3_historical_snapshots.png", dpi=240)
plt.close(fig)
print(t.round(2).to_string(index=False))
print("Matched March/June listings:")
print(panel.round(2).to_string())
