"""Main evaluation on all 16 series (market, 5 Fama-French factors + momentum, 10 industries): 1998-2025, with sub-periods
1998-2015, last 10 years (2016-2025) and last 5 years (2021-2025). Same rules and protocol as market_1998.py, minus the VIX-based
ones (VIX is a market-level quantity).
GARCH: each series gets its own GARCH(1,1) (zero mean, Gaussian QMLE on daily returns in %), re-estimated every January on all
data up to the previous December (expanding window, first fit January 1970), so every forecast is real time. The variance
recursion is computed as one FFT convolution (the same exact filter as ginn/GINN_v2.ipynb). Monthly forecast = sum of the
1- to 21-day-ahead GARCH forecasts made on the last trading day of the previous month. Cached in main_eval/results/.
Rules: Unmanaged; MM on RV; MM on GARCH; Xu switches (GARCH); Conditional quintiles (GARCH); Recovery state (RV).
c: mean exposure 1 on 1970-1997, frozen; w in [0, 2]; 5 bps per unit turnover (factor trading costs are understated).
Tests: mean dSR across series with a paired 6-month block bootstrap whose blocks are common to all series; count of series better."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS
from garch_tools import garch_monthly

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
FAC = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
PERIODS = {"1998-2025 (main)": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"),
           "2016-2025 (last 10y)": ("2016-01", "2025-12"), "2021-2025 (last 5y)": ("2021-01", "2025-12")}
CAL = ("1970-01", "1997-12")
OUT = _P("main_eval/results"); OUT.mkdir(parents=True, exist_ok=True)

mb = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
dd_all = pd.read_csv("data/ff_daily_garchdata_long.csv")
dd_all["date"] = pd.to_datetime(dd_all["Date"], format="%d-%m-%Y")

def rv_states(s, months):
    r = dd_all[dd_all["series"] == s].set_index("date")["ret"].sort_index()
    eom = r.groupby(r.index.to_period("M")).apply(lambda z: z.index[-1])
    f = lambda k: pd.Series(((r ** 2).rolling(k).sum() * 21 / k).loc[eom.values].to_numpy(), index=eom.index.astype(str)).shift(1).reindex(months)
    return f(5), f(63)


def rolling_intercept(r, v, window=60):
    a = pd.Series(np.nan, index=r.index)
    for i in range(window, len(r)):
        y, x = r.iloc[i - window:i].to_numpy(), v.iloc[i - window:i].to_numpy(); ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 48:
            a.iloc[i] = np.polyfit(x[ok], y[ok], 1)[1]
    return a


def mm(v):
    return pd.Series(exposure(v, solve_c(v.loc[CAL[0]:CAL[1]])), index=v.index)


nets, rows = {}, []
for s in SERIES:
    ms = mb[mb["series"] == s].set_index("month").sort_index().loc["1963-07":"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    rv1 = ms["rv"].shift(1); vg = garch_monthly(s).reindex(ms.index); rv5, rv63 = rv_states(s, ms.index)
    W = {"Unmanaged": pd.Series(1.0, index=ms.index), "MM on RV": mm(rv1), "MM on GARCH": mm(vg)}
    hold = e.rolling(12).mean().shift(1) > 0; scale = rolling_intercept(e, rv1) > 0
    W["Xu switches (GARCH)"] = W["MM on GARCH"].where(scale, 1.0).where(hold, 0.0)
    q20, q80 = rv1.expanding(120).quantile(0.2), rv1.expanding(120).quantile(0.8)
    W["Conditional quintiles (GARCH)"] = W["MM on GARCH"].where((rv1 >= q80) | (rv1 <= q20), 1.0)
    W["Recovery state (RV)"] = W["MM on RV"].where(~((rv1 > q80) & (rv5 < rv63)), np.maximum(W["MM on RV"], 1.0))
    for k, w in W.items():
        w = w.loc["1970-01":]
        sim = simulate(w, e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index
        nets[(k, s)] = sim.net_ex
        for plab, (a, b) in PERIODS.items():
            z = sim.net_ex.loc[a:b]; R = e.loc[a:b]; x = w.loc[a:b]; xn = x / x.mean(); wl = (1 + z + rf.loc[a:b]).cumprod()
            rows.append({"series": s, "group": "factor" if s in FAC else "industry", "period": plab, "rule": k,
                         "SR": z.mean() / z.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(), "VT": R.std() / (xn * R).std(),
                         "MaxDD": (wl / wl.cummax() - 1).min()})
    print("done", s, flush=True)

T = pd.DataFrame(rows); T.to_csv(OUT / "panel_1998_by_series.csv", index=False)
RULES = list(dict.fromkeys(T["rule"]))
pd.set_option("display.width", 220)
rng = np.random.default_rng(0)
def srb(M):                                   # Sharpe of each bootstrap sample; 0 if a sample is all cash (sd 0)
    sd = M.std(1, ddof=1)
    return np.where(sd > 0, M.mean(1) / np.where(sd > 0, sd, 1) * np.sqrt(12), 0.0)
for plab, (a, b) in PERIODS.items():
    n = len(nets[("Unmanaged", "Mkt-RF")].loc[a:b]); BB, BL = 4000, 6
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    P = T[T.period == plab].pivot(index="series", columns="rule", values="SR")
    print(f"\n======== {plab}, {n} months. Mean net Sharpe across series (all 16 / 6 factors / 10 industries)")
    summ = pd.DataFrame({"all 16": P.mean(), "factors": P.loc[FAC].mean(), "industries": P.drop(index=FAC).mean()}).loc[RULES]
    print(summ.round(3).to_string())
    for ref in ("Unmanaged", "MM on RV"):
        print(f"  mean dSR vs {ref} [95% CI], series better:")
        for k in RULES:
            if k == ref:
                continue
            for glab, sers in (("all 16", SERIES), ("factors", FAC), ("industries", [x for x in SERIES if x not in FAC])):
                dd = np.mean([srb(nets[(k, s)].loc[a:b].to_numpy()[I]) - srb(nets[(ref, s)].loc[a:b].to_numpy()[I]) for s in sers], axis=0)
                d0 = (P.loc[sers, k] - P.loc[sers, ref]).mean(); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
                print(f"    {k:30s} {glab:10s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]  {int((P.loc[sers, k] > P.loc[sers, ref]).sum())}/{len(sers)}")
print(f"\nPer-series table saved to {OUT / 'panel_1998_by_series.csv'}")
