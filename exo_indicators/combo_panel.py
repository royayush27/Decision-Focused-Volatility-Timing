"""Panel test of vol + expected-return rules on the 16 French series.
w = c * max(mu_hat, 0) / v_hat, v_hat = QMLE GARCH (dfl_diagnosis/panel), mu_hat = expanding OLS of the series'
excess return on one market-level predictor known at the end of t-1; 'combo9' = equal-weight average of the nine
single-predictor forecasts. Benchmark: same rule with the expanding historical mean (Moreira-Muir up to c).
Estimation from 1990-02, OOS from 1995, c calibrated 1995-2004, evaluation 2005-2024, kappa 5 bps."""
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
from calibrate_c import solve_c
from risk_dfl import FACTORS

S = "exo_indicators/"
SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
EST, OOS, CAL, EV = "1990-02", "1995-01", ("1995-01", "2004-12"), ("2005-01", "2024-12")
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
ind = pd.read_csv(S + "indicators_monthly.csv", index_col="month")
_v = pd.read_csv("data/VIXCLS.csv"); _v.columns = ["date", "vix"]
_v["date"] = pd.to_datetime(_v["date"]); _v = _v.dropna()
vix = _v.set_index("date")["vix"].resample("ME").last() ** 2 / 1e4 / 12; vix.index = vix.index.strftime("%Y-%m")
mk = m[m["series"] == "Mkt-RF"].set_index("month").sort_index()

L1, L2 = ind.shift(1), ind.shift(2)
P = pd.DataFrame({"DEF": L1["DEF"], "CS": L1["CS"], "TERM": L1["TERM"], "lEPU": np.log(L1["EPU"]), "NS": L2["NS_avg"],
                  "lTRADE": np.log(L2["EPU_TRADE"]), "lMON": np.log(L2["EPU_MON"]), "lNATSEC": np.log(L2["EPU_NATSEC"])})
P["VRP"] = vix.shift(1).reindex(P.index) - mk["rv"].shift(1).reindex(P.index)
PRED = list(P.columns)


def rule(mu, v):
    sig = np.maximum(mu, 0) / v; sig = sig.where(sig > 0, 1e-12)
    return pd.Series(np.clip(solve_c(1 / sig.loc[CAL[0]:CAL[1]].dropna()) * sig, 0, 2), index=v.index).fillna(1.0)


nets, rows = {}, []
for s in SERIES:
    ms = m[m["series"] == s].set_index("month").sort_index().loc["1970-01":"2025-12"]
    rf = rf_all.reindex(ms.index); ex = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    v = pd.read_csv(f"dfl_diagnosis/panel/{s}_garch.csv", index_col="month")["qmle"].reindex(ms.index)
    Z = P.reindex(ms.index)
    MU = {k: pd.Series(np.nan, index=ms.index) for k in ["hist"] + PRED}
    for t in ms.loc[OOS:"2025-12"].index:
        tr = ms.loc[EST:t].index[:-1]
        MU["hist"][t] = ex.loc[tr].mean()
        for p in PRED:
            ok = Z[p].loc[tr].notna()
            b, a = np.polyfit(Z[p].loc[tr][ok], ex.loc[tr][ok], 1)
            MU[p][t] = a + b * Z[p].loc[t]
    MU["combo9"] = pd.concat([MU[p] for p in PRED], axis=1).mean(axis=1)
    a_, b_ = EV; R = ex.loc[a_:b_]
    for k in ("hist", "combo9", "lEPU", "VRP"):
        w = rule(MU[k], v); sim = simulate(w, ex, rf, 5e-4); sim.index = ms.index
        z = sim.net_ex.loc[a_:b_]; nets[(k, s)] = z.to_numpy()
        rows.append({"series": s, "mu": k, "SR": z.mean() / z.std() * np.sqrt(12),
                     "R2_ret": 1 - ((R - MU[k].loc[a_:b_]) ** 2).sum() / ((R - MU["hist"].loc[a_:b_]) ** 2).sum()})
T = pd.DataFrame(rows)
n = len(nets[("hist", "Mkt-RF")]); BB, BL = 4000, 6
rng = np.random.default_rng(0); st = rng.integers(0, n, (BB, int(np.ceil(n / BL))))
I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
print(f"16 series, eval {EV}; mu benchmark = historical mean with the same GARCH v_hat")
for k in ("combo9", "lEPU", "VRP"):
    dd = np.mean([srb(nets[(k, s)][I]) - srb(nets[("hist", s)][I]) for s in SERIES], axis=0)
    d0 = np.mean([T.query("mu==@k and series==@s").SR.iloc[0] - T.query("mu=='hist' and series==@s").SR.iloc[0] for s in SERIES])
    nb = sum(T.query("mu==@k and series==@s").SR.iloc[0] > T.query("mu=='hist' and series==@s").SR.iloc[0] for s in SERIES)
    print(f"  {k:7s} mean dSR {d0:+.3f} [{np.percentile(dd - dd.mean() + d0, 2.5):+.3f}, {np.percentile(dd - dd.mean() + d0, 97.5):+.3f}] "
          f"p={np.mean(np.abs(dd - dd.mean()) >= abs(d0)):.3f}  better {nb}/16  mean R2_ret {T[T.mu == k].R2_ret.mean():+.4f}")
