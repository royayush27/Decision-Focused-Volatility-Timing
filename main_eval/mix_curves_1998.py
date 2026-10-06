"""Equity and drawdown curves, Sharpe and Sortino for equal-weight factor mixes, vol-managed with MM on GARCH.
Each mix = equal weights in its factors, rebalanced monthly. MM on GARCH is applied to the mix itself: a real-time
GARCH(1,1) on the mix's daily returns (main_eval/garch_tools.py) gives the variance forecast for month t, exposure
w = c / v_hat with c frozen on 1970-1997 (mean exposure 1), w in [0, 2], 5 bps per unit of exposure change.
The unmanaged mix is kept for comparison. Wealth = growth of $1 from January 1998 including the risk-free rate.
'Levered' view: constant leverage that gives 15% annual volatility, set from 1970-1997 data only (real time); the
extra borrowing is assumed at the risk-free rate (optimistic for long-short factors).
Sortino = annualised mean excess return / annualised downside deviation (monthly excess returns below 0).
Writes main_eval/results/mix_curves_monthly.csv (net excess returns) and prints the summary tables."""
import json
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from garch_tools import daily, garch_monthly

MIXES = {"All six": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
         "Mkt + RMW + CMA + Mom": ["Mkt-RF", "RMW", "CMA", "Mom"],
         "Mkt + RMW + Mom": ["Mkt-RF", "RMW", "Mom"],
         "FF5 (no Mom)": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
         "FF3 (Mkt + SMB + HML)": ["Mkt-RF", "SMB", "HML"],
         "No market (SMB + HML + RMW + CMA + Mom)": ["SMB", "HML", "RMW", "CMA", "Mom"],
         "RMW + CMA + Mom": ["RMW", "CMA", "Mom"],
         "Mkt + Mom": ["Mkt-RF", "Mom"],
         "Market alone": ["Mkt-RF"]}
PERIODS = {"1998-2025": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"),
           "2016-2025": ("2016-01", "2025-12"), "2021-2025": ("2021-01", "2025-12")}
CAL = ("1970-01", "1997-12")
OUT = _P("main_eval/results"); OUT.mkdir(parents=True, exist_ok=True)

m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
F = sorted({f for v in MIXES.values() for f in v})
Rm = pd.DataFrame({f: m[m["series"] == f].set_index("month").sort_index()["ret_m"] for f in F}).loc["1963-07":"2025-12"]
d = daily()
Rd = pd.DataFrame({f: d[d["series"] == f].set_index("date")["ret"] for f in F}).sort_index()
rf = rf.reindex(Rm.index)


def stats(z, r):
    total = z + r; wl = np.r_[1.0, (1 + total).cumprod().to_numpy()]
    dn = np.sqrt(np.mean(np.minimum(z, 0) ** 2))
    return {"Sharpe": z.mean() / z.std() * np.sqrt(12), "Sortino": z.mean() * 12 / (dn * np.sqrt(12)),
            "CAGR_%": 100 * (wl[-1] ** (12 / len(z)) - 1), "Vol_%": 100 * z.std() * np.sqrt(12),
            "MaxDD_%": 100 * (wl / np.maximum.accumulate(wl) - 1).min()}


series, lever, rows = {}, {}, []
for name, fs in MIXES.items():
    e = Rm[fs].mean(axis=1)
    tag = "mix_" + "_".join(f.replace("-", "") for f in fs)
    v = garch_monthly(tag, Rd[fs].mean(axis=1).dropna()).reindex(e.index)
    w = pd.Series(exposure(v, solve_c(v.loc[CAL[0]:CAL[1]])), index=e.index).loc["1970-01":]
    sim = simulate(w, e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index
    for ver, z in (("MM-GARCH", sim.net_ex), ("Unmanaged", e.loc["1970-01":])):
        L = 0.15 / (z.loc[CAL[0]:CAL[1]].std() * np.sqrt(12))
        series[(name, ver)] = z; lever[(name, ver)] = L
        for p, (a, b) in PERIODS.items():
            for lv, k in (("as is", 1.0), ("levered to 15% vol", L)):
                rows.append({"mix": name, "version": ver, "period": p, "view": lv, **stats(k * z.loc[a:b], rf.loc[a:b])})
T = pd.DataFrame(rows)
pd.set_option("display.width", 220)
for p in PERIODS:
    for lv in ("as is", "levered to 15% vol"):
        X = T[(T.period == p) & (T.view == lv)].pivot_table(index="mix", columns="version", values=["Sharpe", "Sortino", "CAGR_%", "MaxDD_%"])
        X = X.loc[list(MIXES)]
        print(f"\n===== {p}, {lv}")
        print(X.round(2).to_string())
out = pd.DataFrame({f"{n} | {v}": z.loc["1998-01":"2025-12"] for (n, v), z in series.items()})
out["RF"] = rf.loc["1998-01":"2025-12"]; out.to_csv(OUT / "mix_curves_monthly.csv", index_label="month")
json.dump({"months": list(out.index), "rf": list(out["RF"].round(8)),
           "series": {f"{n}|{v}": list(series[(n, v)].loc["1998-01":"2025-12"].round(8)) for (n, v) in series},
           "lever": {f"{n}|{v}": round(L, 4) for (n, v), L in lever.items()}, "mixes": list(MIXES)},
          open(OUT / "mix_curves.json", "w"))
print("\nleverage to 15% vol (set on 1970-1997):", {f"{n}|{v}": round(L, 2) for (n, v), L in lever.items()})
