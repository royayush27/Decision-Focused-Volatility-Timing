"""Real-time monthly variance forecasts from GARCH(1,1) (main) or GJR-GARCH(1,1) (robustness).

Zero-mean model, Gaussian QMLE on daily returns in percent, re-estimated every January on all data up to the previous
December (expanding window). The variance recursion is linear in the squared shocks given the parameters, so the whole
path is one FFT convolution (exact). Monthly forecast for month m = sum of the 1- to 21-day-ahead forecasts made on the
last trading day of month m-1, in decimal units. Results are cached in results/cache/garch_<name>.csv.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import CACHE

sp = lambda z: np.log1p(np.exp(z))


def unpack(raw, gjr=False):
    """omega > 0; alpha, beta (and gamma/2 for GJR) >= 0 with persistence < 1 (softmax with a leftover share)."""
    if gjr:
        w = np.exp(np.r_[raw[1:4], 0.0]); w = w / w.sum()
        return sp(raw[0]) * 0.1, w[0], w[1], 2 * w[2]          # omega, alpha, beta, gamma
    w = np.exp(np.r_[raw[1:3], 0.0]); w = w / w.sum()
    return sp(raw[0]) * 0.1, w[0], w[1], 0.0


def path(raw, r, h0, gjr=False):
    """h[t] = variance of r[t] given data up to t-1."""
    o, a, b, g = unpack(raw, gjr)
    T = len(r); x = o + (a + g * (r[:-1] < 0)) * r[:-1] ** 2; j = np.arange(T - 1); n = 2 * T
    conv = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(b ** j, n), n)[:T - 1]
    return np.r_[h0, conv + h0 * b ** (j + 1)]


def nll(raw, r, gjr=False):
    h = path(raw, r, r.var(), gjr)
    return 0.5 * np.mean(np.log(h[1:]) + r[1:] ** 2 / h[1:])


def garch_monthly(name, r, first_year, gjr=False):
    """r: daily returns in decimals with a DatetimeIndex. Returns a Series of monthly variance forecasts indexed 'YYYY-MM'."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{'gjr' if gjr else 'garch'}_{name}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="month")["v_hat"]
    r = r.dropna().sort_index() * 100
    first_day = r.groupby(r.index.to_period("M")).apply(lambda z: z.index[0])
    x = np.array([-1.0, -2.0, 1.5] + ([-2.0] if gjr else []))
    out = {}
    for year in range(first_year, 2026):
        tr = r.loc[:f"{year - 1}-12-31"].to_numpy()
        x = minimize(nll, x, args=(tr, gjr), method="L-BFGS-B").x
        o, a, b, g = unpack(x, gjr); phi = a + b + g / 2; vbar = o / (1 - phi) if phi < 1 - 1e-10 else np.nan
        h = pd.Series(path(x, r.loc[:f"{year}-12-31"].to_numpy(), tr.var(), gjr), index=r.loc[:f"{year}-12-31"].index)
        for mth, d0 in first_day.items():
            if mth.year == year:
                h1 = h.loc[d0]                     # forecast for the first day of the month, data to the previous month end
                if not np.isfinite(vbar):          # integrated GARCH (persistence 1 in floating point): flat forecast
                    out[str(mth)] = 21 * h1 / 1e4
                else:
                    out[str(mth)] = sum(vbar + phi ** (k - 1) * (h1 - vbar) for k in range(1, 22)) / 1e4
    v = pd.Series(out, name="v_hat"); v.index.name = "month"; v.to_csv(cache)
    return v
