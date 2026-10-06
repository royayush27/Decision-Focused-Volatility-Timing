"""Two literature-backed attempts to separate crashes from rebounds, fixed in advance.
1. Downside-vol timing (Wang and Yan 2021): scale by last month's downside realised variance (sum of r^2 for r<0)
   instead of total realised variance. Idea: rebounds are upside volatility, so they should not cut exposure.
2. Trend overlay (time-series momentum, Moskowitz-Ooi-Pedersen 2012): hold the vol-managed weight only when the
   series' past 12-month return (t-12..t-1) is positive, otherwise cash. Idea: avoid prolonged bear markets.
Protocol: c frozen on 1970-1999 (mean exposure 1), evaluation 2000-2024, kappa 5 bps, w in [0, 2]."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
d = pd.read_csv("data/ff_daily_garchdata_long.csv")
d["r2d"] = np.where(d["ret"] < 0, d["ret"] ** 2, 0.0)
d["r2u"] = np.where(d["ret"] > 0, d["ret"] ** 2, 0.0)
semi = d.groupby(["series", "month"])[["r2d", "r2u"]].sum()

rows, nets = [], {}
for s in SERIES:
    ms = m[m.series == s].set_index("month").sort_index().loc["1963-07":"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    dn = semi.loc[s]["r2d"].reindex(ms.index)
    trend = ((1 + e).rolling(12).apply(np.prod, raw=True) - 1).shift(1) > 0
    v_rv = ms["rv"].shift(1)                                  # Moreira-Muir: last month's realised variance
    v_dn = dn.shift(1).clip(lower=dn.loc[:"1999-12"].quantile(0.01))  # downside RV, floored (a month with no down days)
    v_g = pd.read_csv(f"dfl_diagnosis/panel/{s}_garch.csv", index_col="month")["qmle"].reindex(ms.index)
    sl = slice("1970-01", "2025-12"); e, rf, trend = e.loc[sl], rf.loc[sl], trend.loc[sl]
    W = {}
    for k, v in (("MM (RV)", v_rv), ("downside RV", v_dn), ("GARCH", v_g)):
        v = v.loc[sl]; W[k] = pd.Series(exposure(v, solve_c(v.loc[:"1999-12"])), index=e.index)
    W["MM (RV) + trend"] = W["MM (RV)"].where(trend, 0.0)
    W["GARCH + trend"] = W["GARCH"].where(trend, 0.0)
    W["trend only"] = pd.Series(1.0, index=e.index).where(trend, 0.0)
    W["unmanaged"] = pd.Series(1.0, index=e.index)
    for k, w in W.items():
        sim = simulate(w, e, rf, 5e-4); sim.index = e.index
        z = sim.net_ex.loc["2000-01":"2024-12"]; nets[(k, s)] = z.to_numpy()
        wl = (1 + z + rf.loc["2000-01":"2024-12"]).cumprod()
        x = w.loc["2000-01":"2024-12"]; x = x / x.mean(); R = e.loc["2000-01":"2024-12"]
        rows.append({"series": s, "rule": k, "SR": z.mean() / z.std() * np.sqrt(12), "MT": (x * R).mean() / R.mean(),
                     "MaxDD": (wl / wl.cummax() - 1).min(), "avgW": w.loc["2000-01":"2024-12"].mean()})
T = pd.DataFrame(rows)
print("Market (Mkt-RF), 2000-2024")
print(T[T.series == "Mkt-RF"].set_index("rule")[["SR", "MT", "MaxDD", "avgW"]].round(3).to_string())
print("\nPanel means over 16 series")
print(T.groupby("rule")[["SR", "MaxDD", "avgW"]].mean().round(3).to_string())
n = len(nets[("MM (RV)", "Mkt-RF")]); BB, BL = 4000, 6
rng = np.random.default_rng(0); st = rng.integers(0, n, (BB, int(np.ceil(n / BL))))
I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
P = T.pivot(index="series", columns="rule", values="SR")
print("\nPaired tests (6-month blocks common across series)")
for A, Bn in (("downside RV", "MM (RV)"), ("MM (RV) + trend", "MM (RV)"), ("GARCH + trend", "GARCH"),
              ("trend only", "unmanaged"), ("GARCH + trend", "unmanaged")):
    for lab, sers in (("market", ["Mkt-RF"]), ("panel", SERIES)):
        dd = np.mean([srb(nets[(A, s)][I]) - srb(nets[(Bn, s)][I]) for s in sers], axis=0)
        d0 = np.mean([P.loc[s, A] - P.loc[s, Bn] for s in sers])
        print(f"  {A:16s} - {Bn:10s} {lab:6s}: dSR {d0:+.3f} [{np.percentile(dd-dd.mean()+d0,2.5):+.3f}, "
              f"{np.percentile(dd-dd.mean()+d0,97.5):+.3f}] p={np.mean(np.abs(dd-dd.mean())>=abs(d0)):.3f} "
              f"better {sum(P.loc[s, A] > P.loc[s, Bn] for s in sers)}/{len(sers)}")
