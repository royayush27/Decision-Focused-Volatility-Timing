"""The volatility-timing rule: exposure w_t = c / v_t, capped, and the trading simulation.

c rules (config.SPECS):
  rematch  every January, c is set on ALL past months that have a forecast, so that the managed series has the same
           volatility as the unmanaged one over those months (with the cap applied). Real time: only data before the year.
  frozen   c is set once on a fixed pre-1998 window so that mean exposure = 1 there (our earlier convention).
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def _clip(x, cap):
    return np.clip(x, 0, np.inf if cap is None else cap)


def c_vol_match(v, f, cap):
    """c such that sd(clip(c / v) * f) = sd(f) on the given months."""
    v, f = np.asarray(v, float), np.asarray(f, float)
    target = f.std(ddof=1)
    if cap is None:
        return target / (f / v).std(ddof=1)
    g = lambda c: (_clip(c / v, cap) * f).std(ddof=1) - target
    lo, hi = 1e-12, target / (f / v).std(ddof=1)
    while g(hi) < 0:
        hi *= 2
    return brentq(g, lo, hi, xtol=1e-16)


def c_mean_one(v, cap):
    """c such that mean(clip(c / v)) = 1 (earlier convention)."""
    v = np.asarray(v, float)
    f = lambda c: _clip(c / v, cap).mean() - 1
    return brentq(f, 0.5 / np.mean(1 / v), (cap or 10) * v.max(), xtol=1e-14)


def mm_weights(v, f, spec, frozen_window=None):
    """v: monthly variance forecast for month t (made at the end of t-1); f: excess return of month t (same index).
    Returns exposures (NaN where no c is available yet) and the c used each month."""
    v, f = v.astype(float), f.reindex(v.index).astype(float)
    cap = spec["cap"]
    if spec["c_rule"] == "frozen":
        a, b = frozen_window
        c = c_mean_one(v.loc[a:b].dropna(), cap)
        cs = pd.Series(c, index=v.index)
    else:
        cs = pd.Series(np.nan, index=v.index)
        ok = v.notna() & f.notna()
        years = sorted({m[:4] for m in v.index[v.notna()]})
        for y in years:
            tr = ok & (v.index < f"{y}-01")
            if tr.sum() >= spec["min_cal_months"]:
                cs[v.index.str[:4] == y] = c_vol_match(v[tr], f[tr], cap)
    return pd.Series(_clip(cs / v, cap), index=v.index), cs


def cap_weights(w, spec):
    return pd.Series(_clip(w, spec["cap"]), index=w.index)


def simulate(w, ex, rf, kappa):
    """Monthly rebalancing to w with drift between rebalances; cost kappa per unit of turnover; returns net excess."""
    w, ex, rf = w.to_numpy(float), ex.to_numpy(float), rf.to_numpy(float)
    R = ex + rf; rp = rf + w * ex
    drift = np.empty_like(w); drift[0] = w[0]
    drift[1:] = w[:-1] * (1 + R[:-1]) / (1 + rp[:-1])
    to = np.abs(w - drift)
    return w * ex - kappa * to, to


def run(w, ex, rf, kappa):
    """Net excess return Series of exposure w on asset ex (months where w is known)."""
    w = w.dropna()
    net, to = simulate(w, ex.reindex(w.index), rf.reindex(w.index), kappa)
    return pd.Series(net, index=w.index), pd.Series(to, index=w.index)


def run_multi(W, R, kappa):
    """Several assets with weights W (rows = months): net excess return, cost on the sum of |weight changes|."""
    W = W.dropna(how="any")
    g = (W * R.reindex(W.index)).sum(axis=1); to = W.diff().abs().sum(axis=1).fillna(0)
    return g - kappa * to, to
