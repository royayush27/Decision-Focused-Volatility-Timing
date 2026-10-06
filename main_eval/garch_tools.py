"""Real-time GARCH(1,1) monthly variance forecasts per series, shared by the main_eval scripts.
Zero-mean GARCH(1,1), Gaussian QMLE on daily returns in %, re-estimated every January on all data up to the previous December
(expanding window, first fit January 1970). The variance recursion is one FFT convolution (exact, as in ginn/GINN_v2.ipynb).
Monthly forecast = sum of the 1- to 21-day-ahead forecasts made on the last trading day of the previous month, in decimal units.
Cached in main_eval/results/garch_<series>.csv; delete a cache file to re-estimate."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT = Path("main_eval/results")
_daily = None


def daily():
    global _daily
    if _daily is None:
        d = pd.read_csv("data/ff_daily_garchdata_long.csv")
        d["date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
        _daily = d
    return _daily


sp = lambda z: np.log1p(np.exp(z))


def unpack(raw):
    w = np.exp(np.r_[raw[1:3], 0.0]); w = w / w.sum()                 # softmax: alpha, beta, leftover (stationary)
    return sp(raw[0]) * 0.1, w[0], w[1]


def garch_path(raw, r, h0):
    o, a, b = unpack(raw); T = len(r); c = o + a * r[:-1] ** 2; j = np.arange(T - 1); n = 2 * T
    conv = np.fft.irfft(np.fft.rfft(c, n) * np.fft.rfft(b ** j, n), n)[:T - 1]
    return np.r_[h0, conv + h0 * b ** (j + 1)]                         # h[t] = variance of r[t] given data to t-1


def nll(raw, r):
    h = garch_path(raw, r, r.var())
    return 0.5 * np.mean(np.log(h[1:]) + r[1:] ** 2 / h[1:])


def garch_monthly(s, r=None, first_year=1970):
    """s = series name in the daily file, or a cache name when a daily return series r (decimal, DatetimeIndex) is given.
    first_year = first January refit (forecasts start that January; training data = everything before it)."""
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"garch_{s}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col="month")["v_hat"]
    r = (daily()[daily()["series"] == s].set_index("date")["ret"] if r is None else r).sort_index() * 100
    first_day = r.groupby(r.index.to_period("M")).apply(lambda z: z.index[0])
    out, x = {}, np.array([-1.0, -2.0, 1.5])
    for year in range(first_year, 2026):
        tr = r.loc[:f"{year - 1}-12-31"].to_numpy()
        x = minimize(nll, x, args=(tr,), method="L-BFGS-B").x
        o, a, b = unpack(x); phi = a + b; vbar = o / (1 - phi)
        h = pd.Series(garch_path(x, r.loc[:f"{year}-12-31"].to_numpy(), tr.var()), index=r.loc[:f"{year}-12-31"].index)
        for mth, d0 in first_day.items():
            if mth.year == year:
                h1 = h.loc[d0]                                          # uses returns up to the last day of the previous month
                if 1 - phi < 1e-10:                                     # integrated GARCH (phi = 1 in floating point): flat forecast
                    out[str(mth)] = 21 * h1 / 1e4
                else:
                    out[str(mth)] = sum(vbar + phi ** (k - 1) * (h1 - vbar) for k in range(1, 22)) / 1e4
    v = pd.Series(out, name="v_hat"); v.index.name = "month"; v.to_csv(cache)
    return v


