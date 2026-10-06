import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backtest import simulate
from calibrate_c import exposure, solve_c
from decision_garch import filter_h, monthly_vhat, persistence, qmle, valid

# Three ways to train the same GARCH / GJR model, annual refits 2000..2025, identical protocol:
#   qmle  : Gaussian quasi-likelihood on daily returns (statistical)
#   util  : realised net utility of w = clip(c/v_hat) on training months (full decision loss, uses returns)
#   risk  : realised variance of the managed position, mean(rv * w^2) with mean(w) = 1 (decision loss on
#           the risk side only; uses realised variance instead of returns, so it carries no mean noise)
# omega fixed at QMLE for the decision losses (scale invariance); forecasts re-anchored to mean training rv.

FACTORS = {"Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"}
KAPPA, GAMMA, W_MAX = 5e-4, 3.0, 2.0
FIRST, LAST = 2000, 2025


def load(series, daily_csv, monthly_csv, rf_csv):
    d = pd.read_csv(daily_csv)
    d = d[d["series"] == series].copy()
    d["Date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
    rd = d.set_index("Date")["ret"].sort_index()
    rf_d = None
    m = pd.read_csv(monthly_csv)
    m = m[m["series"] == series].set_index("month").sort_index().loc["1970-01":"2025-12"]
    rf = pd.read_csv(rf_csv, index_col="month")["RF"].reindex(m.index)
    ex = m["ret_m"] if series in FACTORS else m["ret_m"] - rf      # industries are raw returns
    months = pd.PeriodIndex(m.index, freq="M")
    origin = np.searchsorted(rd.index.values, months.to_timestamp().values) - 1
    return rd.to_numpy() * 100, rd.index, ex, rf, months, origin, m["rv"]


class Window:
    def __init__(self, r, h0, origin, ex, rf, rv, months, y):
        tm = months < pd.Period(f"{y}-01")
        self.r, self.h0, self.origin = r, h0, origin[tm]
        self.ex, self.rf, self.rv = ex[tm], rf[tm], rv[tm].to_numpy()
        self.idx = ex.index[tm]

    def weights(self, th):
        v = monthly_vhat(th, self.r, self.h0, self.origin)
        if not (np.isfinite(v).all() and (v > 0).all()):
            return None
        return exposure(v, solve_c(v, W_MAX), W_MAX)

    def util(self, th):
        w = self.weights(th)
        if w is None:
            return -1e10
        s = simulate(pd.Series(w, index=self.idx), self.ex, self.rf, KAPPA)
        return s.net_ex.mean() * 12 - GAMMA / 2 * s.net_ex.var(ddof=1) * 12

    def risk(self, th):
        w = self.weights(th)
        return 1e10 if w is None else np.mean(self.rv * w ** 2)


def train(q, gjr, f):
    free = [1, 2, 3] if gjr else [1, 3]

    def obj(p):
        th = q.copy(); th[free] = p
        return 1e10 if not valid(th) else f(th)

    res = minimize(obj, q[free], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 800})
    res = minimize(obj, res.x, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 800})
    th = q.copy(); th[free] = res.x
    return th


def run(series, model, data):
    gjr = model == "gjr"
    r, dates, ex, rf, months, origin, rv = data
    out = {k: pd.Series(np.nan, index=ex.index) for k in ("qmle", "util", "risk")}
    diag = []
    for y in range(FIRST, LAST + 1):
        n_tr = np.searchsorted(dates.values, np.datetime64(f"{y}-01-01"))
        h0 = r[:n_tr].var()
        win = Window(r, h0, origin, ex, rf, rv, months, y)
        q = qmle(r[:n_tr], gjr, h0)
        fits = {"qmle": q, "util": train(q, gjr, lambda th: -win.util(th)), "risk": train(q, gjr, win.risk)}
        in_year = (months >= pd.Period(f"{y}-01")) & (months <= pd.Period(f"{y}-12"))
        before = months < pd.Period(f"{y}-01")
        for k, th in fits.items():
            kk = win.rv.mean() / monthly_vhat(th, r, h0, win.origin).mean()
            out[k][in_year] = kk * monthly_vhat(th, r, h0, origin[in_year])
            if y == FIRST:
                out[k][before] = kk * monthly_vhat(th, r, h0, origin[before])
            diag.append({"series": series, "model": model, "year": y, "loss": k, "alpha": th[1], "gamma": th[2],
                         "beta": th[3], "rho": persistence(th)})
    return out, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--series", nargs="+", required=True)
    p.add_argument("--models", nargs="+", default=["garch", "gjr"])
    p.add_argument("--out", type=Path, default=Path("dfl_diagnosis/panel"))
    a = p.parse_args()
    a.out.mkdir(exist_ok=True)
    for s in a.series:
        data = load(s, "data/ff_daily_garchdata_long.csv", "data/ff_monthly_baselinedata.csv", "data/ff_monthly_rf.csv")
        for m in a.models:
            t0 = time.time()
            out, diag = run(s, m, data)
            pd.DataFrame(out).to_csv(a.out / f"{s}_{m}.csv", index_label="month")
            pd.DataFrame(diag).to_csv(a.out / f"{s}_{m}_diag.csv", index=False)
            print(s, m, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
