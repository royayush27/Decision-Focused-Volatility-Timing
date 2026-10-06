"""Is the EPU mean-side result fragile? Sub-periods, timing, smoothing, other start dates."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys, numpy as np, pandas as pd
import exo_test as E, exo_mean as M
from backtest import simulate
from calibrate_c import solve_c
fc, X, ind = E.fc, E.X, E.ind
ex, rf = fc["ret_m"], E.rf
X["lEPU_L2"] = np.log(ind["EPU"].shift(2).reindex(fc.index))
X["lEPU_3m"] = np.log(ind["EPU"].rolling(3).mean().shift(1).reindex(fc.index))
def strat(col, est, oos, cal):
    mu = M.mu_oos(col, est, oos)
    sig = (np.maximum(mu, 0) / M.v); sig = sig.where(sig > 0, 1e-12)
    w = pd.Series(np.clip(solve_c(1 / sig.loc[cal[0]:cal[1]].dropna()) * sig, 0, 2), index=fc.index).fillna(1.0)
    s = simulate(w, ex, rf, 5e-4); s.index = fc.index; return s.net_ex, mu
sr = lambda z: z.mean() / z.std() * np.sqrt(12)
for est, oos, cal in [("1986-03", "1995-01", ("1995-01", "2004-12")), ("1986-03", "1992-01", ("1992-01", "1999-12"))]:
    print(f"\nest {est}, OOS {oos}, c {cal}")
    z0, _ = strat(None, est, oos, cal)
    for col in ["lEPU", "lEPU_L2", "lEPU_3m"]:
        z1, mu = strat(col, est, oos, cal)
        out = []
        for a, b in [("2000-01", "2009-12"), ("2010-01", "2024-12"), ("2005-01", "2014-12"), ("2015-01", "2024-12"), ("2000-01", "2024-12")]:
            if a < cal[1]: 
                if a <= cal[1][:4] + "-12" and cal[1] >= a: pass
            out.append(f"{a[:4]}-{b[:4]}: {sr(z1.loc[a:b]) - sr(z0.loc[a:b]):+.3f}")
        print(f"  {col:8s} dSR vs MM  " + "  ".join(out))
