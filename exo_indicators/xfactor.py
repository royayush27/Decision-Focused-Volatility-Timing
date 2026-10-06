"""Two-layer factor portfolio: (1) timing = scale by forecast variance, (2) allocation = weights across factors.
Six French factors (Mkt-RF, SMB, HML, RMW, CMA, Mom). Forecasts: per-factor GARCH v_hat from dfl_diagnosis/panel,
trained three ways (qmle, risk, util). Correlations: rolling 60-month sample correlation of past returns.
Schemes (all use only information at the end of t-1):
  EW              1/6 each, no timing
  EW vol-managed  each factor scaled by c_i / v_i (c_i frozen on 1970-1999), then 1/6 each   [layer 1 only]
  InvVar          w_i proportional to 1/v_i, sum 1, no timing                                 [layer 2 only]
  GMV             minimum-variance weights from Sigma = D C D, D = sqrt(v_hat), C = 60m corr   [layer 2 only]
  GMV timed       GMV portfolio scaled by c / (w' Sigma w), c frozen on 1970-1999              [layers 1 + 2]
Evaluation 2000-2024, gross and net of 5 bps per unit of weight turnover (factor costs are understated)."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd

from calibrate_c import exposure, solve_c

F = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
R = pd.DataFrame({f: m[m.series == f].set_index("month").sort_index()["ret_m"] for f in F}).loc["1965-01":"2025-12"]
V = {k: pd.DataFrame({f: pd.read_csv(f"dfl_diagnosis/panel/{f}_garch.csv", index_col="month")[k] for f in F}).reindex(R.index)
     for k in ("qmle", "risk", "util")}
C60 = {t: R.loc[:t].iloc[-61:-1].corr().to_numpy() for t in R.loc["1970-01":].index}   # past 60 months, excludes t
idx = R.loc["1970-01":].index


def gmv(vrow, C):
    d = np.sqrt(vrow); S = C * np.outer(d, d); x = np.linalg.solve(S, np.ones(len(d)))
    w = x / x.sum(); return w, w @ S @ w


def weights(scheme, k="qmle"):
    v = V[k].loc[idx]; W = pd.DataFrame(index=idx, columns=F, dtype=float)
    if scheme == "EW":
        W[:] = 1 / 6
    elif scheme == "EW vol-managed":
        for f in F:
            W[f] = exposure(v[f], solve_c(v[f].loc[:"1999-12"])) / 6
    elif scheme == "InvVar":
        W[:] = (1 / v).div((1 / v).sum(axis=1), axis=0)
    elif scheme in ("GMV", "GMV timed"):
        pv = pd.Series(index=idx, dtype=float)
        for t in idx:
            w, p = gmv(v.loc[t].to_numpy(), C60[t]); W.loc[t] = w; pv[t] = p
        if scheme == "GMV timed":
            W = W.mul(pd.Series(exposure(pv, solve_c(pv.loc[:"1999-12"])), index=idx), axis=0)
    return W


def perf(W, kappa=5e-4):
    g = (W * R.loc[idx]).sum(axis=1)
    to = W.diff().abs().sum(axis=1).fillna(0)
    n = g - kappa * to
    out = {}
    for lab, z in (("SR_gross", g), ("SR_net", n)):
        z = z.loc["2000-01":"2024-12"]; out[lab] = z.mean() / z.std() * np.sqrt(12)
    z = n.loc["2000-01":"2024-12"]; wl = (1 + z).cumprod()
    out.update({"vol": z.std() * np.sqrt(12), "MaxDD": (wl / wl.cummax() - 1).min(),
                "gross_exposure": W.abs().sum(axis=1).loc["2000-01":"2024-12"].mean(),
                "turnover": to.loc["2000-01":"2024-12"].mean()})
    return out, n.loc["2000-01":"2024-12"]


rows, nets = {}, {}
for s in ("EW", "EW vol-managed", "InvVar", "GMV", "GMV timed"):
    rows[s], nets[s] = perf(weights(s))
T = pd.DataFrame(rows).T
print("Six factors, 2000-2024 (QMLE GARCH forecasts)")
print(T.round(3).to_string())

# layer-2 DFL question: does the training loss of the variance forecasts matter once they set cross-factor weights?
rows2, nets2 = {}, {}
for s in ("EW vol-managed", "InvVar", "GMV", "GMV timed"):
    for k in ("qmle", "risk", "util"):
        rows2[(s, k)], nets2[(s, k)] = perf(weights(s, k))
T2 = pd.DataFrame(rows2).T[["SR_net", "vol", "MaxDD", "turnover"]]
print("\nSame schemes, variance forecasts trained by QMLE vs decision losses")
print(T2.round(3).to_string())

rng = np.random.default_rng(0)
def boot(a, b, BB=4000, BL=6):
    a, b = a.to_numpy(), b.to_numpy(); n = len(a)
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    sr = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
    d = sr(a[I]) - sr(b[I]); d0 = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * np.sqrt(12)
    return f"{d0:+.3f} [{np.percentile(d-d.mean()+d0,2.5):+.3f}, {np.percentile(d-d.mean()+d0,97.5):+.3f}] p={np.mean(np.abs(d-d.mean())>=abs(d0)):.3f}"
print("\nNet dSR (95% block-bootstrap CI)")
for a in ("EW vol-managed", "InvVar", "GMV", "GMV timed"):
    print(f"  {a:15s} - EW: {boot(nets[a], nets['EW'])}")
print(f"  GMV timed       - GMV: {boot(nets['GMV timed'], nets['GMV'])}")
for s in ("InvVar", "GMV"):
    for k in ("risk", "util"):
        print(f"  {s} {k:4s} - {s} qmle: {boot(nets2[(s, k)], nets2[(s, 'qmle')])}")
