"""Can exogenous indicators fix the MEAN side instead? w = c * max(mu_hat, 0) / v_hat.
mu_hat from expanding OLS of next-month excess return on one indicator (known at end of t-1).
Benchmark: mu_hat = expanding historical mean, which is Moreira-Muir up to the constant c.
Campbell-Thompson OOS R2 for returns is reported too."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd

import exo_test as E
from backtest import simulate
from calibrate_c import exposure, solve_c

fc, X, rf = E.fc, E.X, E.rf
ex, rv = fc["ret_m"], fc["rv"]
base, s2 = E.oos([], "1970-01", "1985-01")
v = np.exp(base + s2 / 2)                        # baseline variance forecast (GARCH + lagged RV regression)


def mu_oos(col, est_start, oos_start):
    mu = pd.Series(np.nan, index=fc.index)
    for t in fc.loc[oos_start:"2025-12"].index:
        tr = fc.loc[est_start:t].index[:-1]
        if col is None:
            mu[t] = ex.loc[tr].mean(); continue
        z = X[col].loc[tr]; r = ex.loc[tr]; ok = z.notna() & r.notna()
        if ok.sum() < 48 or pd.isna(X[col].loc[t]):
            continue
        b, a = np.polyfit(z[ok], r[ok], 1)
        mu[t] = a + b * X[col].loc[t]
    return mu


def run(cands, est_start, oos_start, cal, ev, label):
    print(f"\n===== {label}: mu estimated from {est_start}, OOS from {oos_start}, c on {cal}, eval {ev}")
    a, b = ev
    hist = mu_oos(None, est_start, oos_start)
    rows, nets = [], {}
    for k in [None] + cands:
        mu = hist if k is None else mu_oos(k, est_start, oos_start)
        R = ex.loc[a:b]
        r2 = 1 - ((R - mu.loc[a:b]) ** 2).sum() / ((R - hist.loc[a:b]) ** 2).sum()
        sig = np.maximum(mu, 0) / v                 # Campbell-Thompson style: no negative mean, no shorting
        sig = sig.where(sig > 0, 1e-12)
        w = pd.Series(np.clip(solve_c(1 / sig.loc[cal[0]:cal[1]].dropna()) * sig, 0, 2), index=fc.index).fillna(1.0)
        sim = simulate(w, ex, rf, 5e-4); sim.index = fc.index
        W = w.loc[a:b]; x = W / W.mean(); z = sim.net_ex.loc[a:b]; nets[k or "hist mean (=MM)"] = z.to_numpy()
        rows.append({"mu from": k or "hist mean (=MM)", "R2oos_returns": r2, "MT": (x * R).mean() / R.mean(),
                     "VR": (rv.loc[a:b] * x ** 2).mean() / rv.loc[a:b].mean(), "zero_exposure_share": (W < 1e-6).mean(),
                     "SR_net": z.mean() / z.std() * np.sqrt(12)})
    T = pd.DataFrame(rows).set_index("mu from")
    un = ex.loc[a:b]; print("unmanaged SR:", round(un.mean() / un.std() * np.sqrt(12), 3))
    print(T.round(3).to_string())
    rng = np.random.default_rng(0); n = len(nets["hist mean (=MM)"]); BB, BL = 4000, 6
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    sr = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12); z0 = nets["hist mean (=MM)"]
    for k in cands:
        d = sr(nets[k][I]) - sr(z0[I]); d0 = (nets[k].mean() / nets[k].std(ddof=1) - z0.mean() / z0.std(ddof=1)) * np.sqrt(12)
        print(f"  {k}: dSR vs MM {d0:+.3f} [{np.percentile(d - d.mean() + d0, 2.5):+.3f}, {np.percentile(d - d.mean() + d0, 97.5):+.3f}]")


if __name__ == "__main__":
    run(["DEF", "CS", "TERM"], "1970-01", "1985-01", ("1985-01", "1999-12"), ("2000-01", "2024-12"), "Mean side, spreads")
    run(["DEF", "CS", "TERM", "lEPU", "NS", "lVIX2"], "1986-03", "1995-01", ("1995-01", "2004-12"), ("2005-01", "2024-12"), "Mean side, news era")
