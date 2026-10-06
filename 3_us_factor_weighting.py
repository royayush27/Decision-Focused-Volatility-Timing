"""3. US, how to weight the six factors (Mkt-RF, SMB, HML, RMW, CMA, Mom), with and without timing the whole mix.

Allocation (real time, information at the end of t-1):
  EW                  1/6 each
  Inverse variance    w_i ~ 1 / v_i (GARCH variances)
  Min variance (GMV)  Sigma = D C D, D = GARCH vols, C = correlation over the past 60 months
  Tangency long-only  w ~ max(Sigma^-1 mu_hat, 0), mu_hat = expanding mean since 1963
Timing: none, or scale the whole mix by c / (w' Sigma w) (the spec's c rule and cap). Also 'EW of vol-managed factors':
each factor first managed on its own GARCH, then 1/6 each. Costs on the sum of absolute weight changes.
Benchmarks: the market alone and EW (unmanaged).
"""
import numpy as np
import pandas as pd

import config
from core import data, report
from core.garch import garch_monthly
from core.rule import mm_weights, run_multi
from core.stats import perf, boot_idx, dsr_test, fmt

S = report.start("3_us_factor_weighting", "3. US, how to weight the six factors")
FW = ("1970-01", "1997-12")
F = data.FACTORS
R = data.us_monthly(F); rf = data.rf().reindex(R.index); D = data.us_daily(F)
V = pd.DataFrame({f: garch_monthly(f"us_{f}", D[f].dropna(), config.US_FIRST_GARCH_YEAR) for f in F}).reindex(R.index)
idx = R.loc["1970-01":].index
C60 = {t: R.loc[:t].iloc[-61:-1].corr().to_numpy() for t in idx}
MU = R.expanding().mean().shift(1)
sigma = lambda t: C60[t] * np.outer(np.sqrt(V.loc[t].to_numpy()), np.sqrt(V.loc[t].to_numpy()))


def alloc(name):
    W = pd.DataFrame(index=idx, columns=F, dtype=float)
    for t in idx:
        if name == "EW":
            w = np.ones(6)
        elif name == "Inverse variance":
            w = 1 / V.loc[t].to_numpy()
        elif name == "Min variance (GMV)":
            w = np.linalg.solve(sigma(t), np.ones(6))
        else:
            w = np.maximum(np.linalg.solve(sigma(t), MU.loc[t].to_numpy()), 0)
            w = np.ones(6) if w.sum() <= 0 else w
        W.loc[t] = w / np.abs(w).sum()
    return W


def timed(W):
    pv = pd.Series([W.loc[t].to_numpy() @ sigma(t) @ W.loc[t].to_numpy() for t in idx], index=idx)
    x, _ = mm_weights(pv, (W * R.loc[idx]).sum(axis=1), S, FW)
    return W.mul(x, axis=0)


Wd = {}
for a in ("EW", "Inverse variance", "Min variance (GMV)", "Tangency (long-only)"):
    Wd[a] = alloc(a); Wd[a + " + timing"] = timed(Wd[a])
Wd["EW of vol-managed factors"] = pd.DataFrame({f: mm_weights(V[f].loc[idx], R[f].loc[idx], S, FW)[0] for f in F}) / 6
Wd["Market alone"] = pd.DataFrame(0.0, index=idx, columns=F).assign(**{"Mkt-RF": 1.0})
nets = {k: run_multi(W, R, S["cost"]) for k, W in Wd.items()}
start = max(z.index[0] for z, _ in nets.values())
rows = []
for p, (a, b) in config.PERIODS.items():
    T = {}; I = boot_idx(len(nets["EW"][0].loc[a:b]))
    for k, (z, to) in nets.items():
        z = z.loc[a:b]; T[k] = {**perf(z, rf.loc[a:b]), "gross_exposure": Wd[k].loc[a:b].abs().sum(axis=1).mean(),
                                "turnover_per_yr": 12 * to.loc[a:b].mean()}
        for ref in ("Market alone", "EW"):
            if k != ref:
                d, lo, hi, pv, _ = dsr_test(z, nets[ref][0].loc[a:b], I); T[k][f"dSR vs {ref} [95% CI]"] = fmt(d, lo, hi)
        rows.append({"period": p, "strategy": k, **T[k]})
    print(f"======== {p}"); report.show(pd.DataFrame(T).T)
pd.DataFrame(rows).to_csv(config.out_dir() / "3_us_factor_weighting.csv", index=False)
print("Average weights 1998-2025 (before timing)")
report.show(pd.DataFrame({k: Wd[k].loc["1998-01":"2025-12"].mean() for k in ("EW", "Inverse variance", "Min variance (GMV)", "Tangency (long-only)")}).T)
nn = {}
for a in ("EW", "Inverse variance", "Min variance (GMV)", "Tangency (long-only)"):
    nn[("US factor weighting", a, "Unmanaged")] = nets[a][0]; nn[("US factor weighting", a, "Timed whole mix")] = nets[a + " + timing"][0]
nn[("US factor weighting", "EW", "EW of vol-managed factors")] = nets["EW of vol-managed factors"][0]
report.save_nets("3_us_factor_weighting", nn)
