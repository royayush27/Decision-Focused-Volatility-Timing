import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest import simulate
from calibrate_c import exposure, solve_c
from decision_garch import monthly_vhat, persistence, qmle, valid
from risk_dfl import load, GAMMA, W_MAX

# Cost-aware risk-only decision loss: with exposure normalised to mean 1, utility = mu - gamma/2 var - costs,
# and mu does not depend on theta if expected returns are flat. So minimise
#     gamma/2 * 12 * mean(rv * w^2)  +  12 * kappa * mean(turnover)
# No realised returns enter except through the weight drift, so the loss carries almost no mean noise.

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]


def run(series, kappa):
    r, dates, ex, rf, months, origin, rv = load(series, "data/ff_daily_garchdata_long.csv",
                                               "data/ff_monthly_baselinedata.csv", "data/ff_monthly_rf.csv")
    out = {k: pd.Series(np.nan, index=ex.index) for k in ("qmle", "cost_risk")}
    for y in range(2000, 2026):
        n_tr = np.searchsorted(dates.values, np.datetime64(f"{y}-01-01"))
        h0 = r[:n_tr].var()
        tm = months < pd.Period(f"{y}-01")
        org, ext, rft, rvt, idx = origin[tm], ex[tm], rf[tm], rv[tm].to_numpy(), ex.index[tm]

        def loss(th):
            if not valid(th):
                return 1e10
            v = monthly_vhat(th, r, h0, org)
            if not (np.isfinite(v).all() and (v > 0).all()):
                return 1e10
            w = exposure(v, solve_c(v, W_MAX), W_MAX)
            s = simulate(pd.Series(w, index=idx), ext, rft, kappa)
            return GAMMA / 2 * 12 * np.mean(rvt * w ** 2) + 12 * kappa * s.turnover.mean()

        q = qmle(r[:n_tr], False, h0)
        obj = lambda p: loss(np.r_[q[0], p[0], 0.0, p[1]])
        res = minimize(obj, q[[1, 3]], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-11, "maxiter": 800})
        res = minimize(obj, res.x, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-11, "maxiter": 800})
        th = np.r_[q[0], res.x[0], 0.0, res.x[1]]
        te = (months >= pd.Period(f"{y}-01")) & (months <= pd.Period(f"{y}-12"))
        for k, t_ in (("qmle", q), ("cost_risk", th)):
            kk = rvt.mean() / monthly_vhat(t_, r, h0, org).mean()
            out[k][te] = kk * monthly_vhat(t_, r, h0, origin[te])
            if y == 2000:
                out[k][tm] = kk * monthly_vhat(t_, r, h0, org)
    return pd.DataFrame(out)


if __name__ == "__main__":
    kappa = float(sys.argv[1])
    sers = sys.argv[2:] or SERIES
    for s in sers:
        t0 = time.time()
        run(s, kappa).to_csv(f"dfl_diagnosis/panel/{s}_cost{int(kappa*1e4)}.csv", index_label="month")
        print(s, kappa, f"{time.time()-t0:.0f}s", flush=True)
