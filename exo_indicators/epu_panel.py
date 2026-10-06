"""Pre-registered panel test of the one lead: policy uncertainty (EPU) on the MEAN side.
Rule A (Moreira-Muir): w = c / v_hat.   Rule B: w = c * max(mu_hat, 0) / v_hat,
mu_hat = expanding OLS of the series' excess return on log EPU (lag 1 = month t-1 average of daily EPU).
v_hat = QMLE GARCH forecast from dfl_diagnosis/panel (annual refits). 16 French series, kappa 5 bps.
Common 6-month blocks across series for the bootstrap."""
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
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
ind = pd.read_csv(S + "indicators_monthly.csv", index_col="month")
LAG = int(sys.argv[1]) if len(sys.argv) > 1 else 1
EST, OOS, CAL, EV = "1986-03", "1995-01", ("1995-01", "2004-12"), ("2005-01", "2024-12")
if len(sys.argv) > 2:
    OOS, CAL, EV = "1992-01", ("1992-01", "1999-12"), ("2000-01", "2024-12")


def rule(mu, v, cal):
    sig = (np.maximum(mu, 0) / v); sig = sig.where(sig > 0, 1e-12)
    return pd.Series(np.clip(solve_c(1 / sig.loc[cal[0]:cal[1]].dropna()) * sig, 0, 2), index=v.index).fillna(1.0)


nets, rows = {}, []
for s in SERIES:
    ms = m[m["series"] == s].set_index("month").sort_index().loc["1970-01":"2025-12"]
    rf = rf_all.reindex(ms.index)
    ex = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    v = pd.read_csv(f"dfl_diagnosis/panel/{s}_garch.csv", index_col="month")["qmle"].reindex(ms.index)
    z = np.log(ind["EPU"].shift(LAG)).reindex(ms.index)
    mu_h, mu_e = pd.Series(np.nan, index=ms.index), pd.Series(np.nan, index=ms.index)
    for t in ms.loc[OOS:"2025-12"].index:
        tr = ms.loc[EST:t].index[:-1]
        mu_h[t] = ex.loc[tr].mean()
        ok = z.loc[tr].notna()
        b, a = np.polyfit(z.loc[tr][ok], ex.loc[tr][ok], 1)
        mu_e[t] = a + b * z.loc[t]
    for k, mu in (("MM", mu_h), ("EPU", mu_e)):
        w = rule(mu, v, CAL)
        sim = simulate(w, ex, rf, 5e-4); sim.index = ms.index
        a_, b_ = EV
        W = w.loc[a_:b_]; x = W / W.mean(); R = ex.loc[a_:b_]; net = sim.net_ex.loc[a_:b_]
        nets[(k, s)] = net.to_numpy()
        rows.append({"series": s, "rule": k, "SR_net": net.mean() / net.std() * np.sqrt(12),
                     "MT": (x * R).mean() / R.mean(), "R2_ret": np.nan if k == "MM" else
                     1 - ((R - mu_e.loc[a_:b_]) ** 2).sum() / ((R - mu_h.loc[a_:b_]) ** 2).sum()})
T = pd.DataFrame(rows).pivot(index="series", columns="rule", values=["SR_net", "MT", "R2_ret"])
T = T.drop(columns=("R2_ret", "MM"))
T["dSR"] = T[("SR_net", "EPU")] - T[("SR_net", "MM")]
print(f"EPU lag {LAG}; mu from {EST}; OOS {OOS}; c on {CAL}; eval {EV}")
print(T.round(3).to_string())
n = len(next(iter(nets.values()))); BB, BL = 4000, 6
rng = np.random.default_rng(0); st = rng.integers(0, n, (BB, int(np.ceil(n / BL))))
I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
sr = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
d = np.mean([sr(nets[("EPU", s)][I]) - sr(nets[("MM", s)][I]) for s in SERIES], axis=0)
d0 = T["dSR"].mean()
print(f"\npanel mean dSR (EPU rule - MM) {d0:+.3f}  95% CI [{np.percentile(d - d.mean() + d0, 2.5):+.3f}, "
      f"{np.percentile(d - d.mean() + d0, 97.5):+.3f}]  p={np.mean(np.abs(d - d.mean()) >= abs(d0)):.3f}  "
      f"series better {int((T['dSR'] > 0).sum())}/16  mean R2_ret {T[('R2_ret', 'EPU')].mean():+.4f}")
