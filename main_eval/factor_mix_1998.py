"""Can a MIX of the six factors (Mkt-RF, SMB, HML, RMW, CMA, Mom) do well, and does vol timing help inside the mix?
Two layers, all real time (information at the end of t-1; parameters frozen before 1998 where they are calibrated):
  allocation (which factors):  EW 1/6; inverse variance (w_i ~ 1/v_i); minimum variance (GMV) with Sigma = D C D,
      D = sqrt(GARCH variances), C = past 60-month correlation; tangency long-only (w ~ max(Sigma^-1 mu_hat, 0),
      mu_hat = expanding mean of each factor since 1963, normalised to sum 1)
  timing (how much in total):  none, or scale the whole mix by c / (w' Sigma w) with c frozen on 1970-1997 (mean exposure 1)
Also 'EW of vol-managed factors': each factor scaled by its own c_i / v_i first (timing before allocation, Moreira-Muir style).
GARCH variances from main_eval/garch_tools.py. Gross weights sum to 1 before timing; timed exposure capped at 2.
Costs 5 bps per unit of factor-weight turnover (understated for factors). Evaluation 1998-2025 and sub-periods.
Benchmarks: the market alone (unmanaged) and EW. Paired 6-month block bootstrap."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from calibrate_c import exposure, solve_c
from garch_tools import garch_monthly

F = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
PERIODS = {"1998-2025 (main)": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"),
           "2016-2025 (last 10y)": ("2016-01", "2025-12"), "2021-2025 (last 5y)": ("2021-01", "2025-12")}
CAL = ("1970-01", "1997-12")
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
R = pd.DataFrame({f: m[m["series"] == f].set_index("month").sort_index()["ret_m"] for f in F}).loc["1963-07":"2025-12"]
V = pd.DataFrame({f: garch_monthly(f) for f in F}).reindex(R.index)
idx = R.loc["1970-01":].index
C60 = {t: R.loc[:t].iloc[-61:-1].corr().to_numpy() for t in idx}            # past 60 months, excludes t
MU = R.expanding().mean().shift(1)


def sigma(t):
    d = np.sqrt(V.loc[t].to_numpy()); return C60[t] * np.outer(d, d)


def alloc(name):
    W = pd.DataFrame(index=idx, columns=F, dtype=float)
    for t in idx:
        S = sigma(t)
        if name == "EW":
            w = np.ones(6)
        elif name == "Inverse variance":
            w = 1 / V.loc[t].to_numpy()
        elif name == "Min variance (GMV)":
            w = np.linalg.solve(S, np.ones(6))
        else:                                                               # tangency, long-only
            w = np.maximum(np.linalg.solve(S, MU.loc[t].to_numpy()), 0)
            if w.sum() <= 0:
                w = np.ones(6)
        W.loc[t] = w / np.abs(w).sum()
    return W


def timed(W):
    pv = pd.Series([W.loc[t].to_numpy() @ sigma(t) @ W.loc[t].to_numpy() for t in idx], index=idx)
    return W.mul(pd.Series(exposure(pv, solve_c(pv.loc[CAL[0]:CAL[1]])), index=idx), axis=0)


S = {}
for a in ("EW", "Inverse variance", "Min variance (GMV)", "Tangency (long-only)"):
    W = alloc(a); S[a] = W; S[a + " + vol timing"] = timed(W)
VM = pd.DataFrame({f: exposure(V[f].loc[idx], solve_c(V[f].loc[CAL[0]:CAL[1]])) for f in F}, index=idx) / 6
S["EW of vol-managed factors"] = VM
S["Market alone"] = pd.DataFrame(0.0, index=idx, columns=F).assign(**{"Mkt-RF": 1.0})

nets = {}
for k, W in S.items():
    g = (W * R.loc[idx]).sum(axis=1); to = W.diff().abs().sum(axis=1).fillna(0)
    nets[k] = (g - 5e-4 * to, W, to)
pd.set_option("display.width", 220)
rng = np.random.default_rng(0)
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
for plab, (a, b) in PERIODS.items():
    rows = {}
    for k, (z, W, to) in nets.items():
        zz = z.loc[a:b]; wl = (1 + zz).cumprod()
        rows[k] = {"SR_net": zz.mean() / zz.std() * np.sqrt(12), "ann_ret_%": 1200 * zz.mean(), "ann_vol_%": 100 * zz.std() * np.sqrt(12),
                   "MaxDD": (wl / wl.cummax() - 1).min(), "gross_exposure": W.loc[a:b].abs().sum(axis=1).mean(),
                   "turnover_per_yr": 12 * to.loc[a:b].mean()}
    T = pd.DataFrame(rows).T
    n = len(nets["EW"][0].loc[a:b]); BB, BL = 4000, 6
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    for ref, col in (("Market alone", "dSR vs market [95% CI]"), ("EW", "dSR vs EW [95% CI]")):
        out = []
        for k in T.index:
            if k == ref:
                out.append(""); continue
            A, B = nets[k][0].loc[a:b].to_numpy(), nets[ref][0].loc[a:b].to_numpy()
            d0 = T.loc[k, "SR_net"] - T.loc[ref, "SR_net"]; dd = srb(A[I]) - srb(B[I]); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
            out.append(f"{d0:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        T[col] = out
    print(f"\n======== {plab}, {n} months")
    print(T.round(3).to_string())
print("\nAverage weights 1998-2025 (before timing):")
print(pd.DataFrame({k: S[k].loc["1998-01":"2025-12"].mean() for k in ("EW", "Inverse variance", "Min variance (GMV)", "Tangency (long-only)")}).T.round(3).to_string())
