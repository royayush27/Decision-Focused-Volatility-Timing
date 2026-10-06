# archived in old/scipy_garch: run from anywhere, paths below are relative to the project root
import os as _os, sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
for _p in (_ROOT, _P(__file__).resolve().parent):
    _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from garch_x import load_inputs, variance_path, fit

U = ""
mf = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")
fc = {
    "GARCH-X (VIX)": pd.read_csv("old/scipy_garch/forecasts_vix/garchx_vix_stat.csv", index_col="month")["v_hat"],
    "GARCH (same window)": pd.read_csv("old/scipy_garch/forecasts_vix/garch_1990_stat.csv", index_col="month")["v_hat"],
    "GARCH (1970 fit)": pd.read_csv("results_1970/forecasts/garch_stat.csv", index_col="month")["v_hat"],
    "GJR (1970 fit)": pd.read_csv("results_1970/forecasts/gjr_stat.csv", index_col="month")["v_hat"],
    "Lagged RV": mf["rv_baseline_forecast"],
}
rv = mf["rv"]
vol = lambda s: np.sqrt(s * 12) * 100          # monthly variance -> annualised vol, %
style = {"GARCH-X (VIX)": ("#d62728", 2.4), "GARCH (same window)": ("#1f77b4", 1.6),
         "GARCH (1970 fit)": ("#7fb3e6", 1.2), "GJR (1970 fit)": ("#2ca02c", 1.4), "Lagged RV": ("#888888", 1.4)}

events = {"Aug 2007 to Mar 2009": ("2007-06", "2009-06"), "Aug 2011 US downgrade": ("2011-04", "2012-01"),
          "Feb 2018 Volmageddon": ("2017-10", "2018-08"), "Covid 2020": ("2019-11", "2020-12"),
          "Aug 2024 carry unwind": ("2024-04", "2024-12"), "Apr 2025 tariffs": ("2024-12", "2025-08")}
fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
for ax, (title, (a, b)) in zip(axes.ravel(), events.items()):
    idx = pd.period_range(a, b, freq="M").astype(str)
    ax.plot(range(len(idx)), vol(rv.reindex(idx)), color="black", lw=2.2, ls="--", label="Realised (next month)")
    for k, s in fc.items():
        c, w = style[k]
        ax.plot(range(len(idx)), vol(s.reindex(idx)), color=c, lw=w, label=k)
    ax.set_xticks(range(0, len(idx), 2)); ax.set_xticklabels(idx[::2], rotation=45, fontsize=8)
    ax.set_title(title, fontsize=11); ax.set_ylabel("annualised vol, %"); ax.grid(alpha=0.25)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False)
fig.suptitle("Monthly forecasts (made at start of month) vs realised volatility that month", fontsize=13)
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig("old/scipy_garch/figures/vix_monthly_events.png", dpi=130)

# daily reaction: one-day-ahead conditional vol, full-sample fits (descriptive, in-sample)
r, x = load_inputs("data/ff_daily_garchdata_long.csv", "data/VIXCLS.csv")
rr, xx = r.to_numpy(), x.to_numpy()
thx, thg = fit(rr, xx, True).x, fit(rr, xx, False).x
print("full-sample GARCH-X", thx.round(4), " GARCH", thg.round(4))
hx, hg = variance_path(thx, rr, xx), variance_path(thg, rr, xx)
real21 = pd.Series(rr ** 2, index=r.index).rolling(21).mean()
d = pd.DataFrame({"GARCH-X (VIX)": np.sqrt(hx * 252), "GARCH": np.sqrt(hg * 252),
                  "VIX": np.sqrt(xx * 252), "|r| (daily, x16)": np.abs(rr) * np.sqrt(252)}, index=r.index)
wins = {"Lehman, Sep to Dec 2008": ("2008-08-15", "2008-12-31"), "Covid, Feb to Apr 2020": ("2020-02-01", "2020-05-01"),
        "Volmageddon, Jan to Mar 2018": ("2018-01-15", "2018-03-31")}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (t, (a, b)) in zip(axes, wins.items()):
    w = d.loc[a:b]
    ax.bar(w.index, w["|r| (daily, x16)"], color="#dddddd", width=1.0, label="|daily return| annualised")
    ax.plot(w.index, w["VIX"], color="#9467bd", lw=1.3, ls=":", label="VIX")
    ax.plot(w.index, w["GARCH"], color="#1f77b4", lw=2, label="GARCH, 1-day-ahead vol")
    ax.plot(w.index, w["GARCH-X (VIX)"], color="#d62728", lw=2, label="GARCH-X, 1-day-ahead vol")
    ax.set_title(t); ax.set_ylabel("annualised, %"); ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_ylim(0, min(w.max().max(), 200) * 1.05)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
fig.suptitle("Daily reaction speed (full-sample parameters, descriptive)", fontsize=13)
fig.tight_layout(rect=(0, 0.08, 1, 0.95)); fig.savefig("old/scipy_garch/figures/vix_daily_reaction.png", dpi=130)

# numbers: how fast does each model move after the shock days
for t, (a, b) in wins.items():
    w = d.loc[a:b]; pk = w["VIX"].idxmax()
    print(t, "VIX peak", pk.date(), round(w.loc[pk, "VIX"], 1))
