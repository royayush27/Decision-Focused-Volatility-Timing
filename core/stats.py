"""Statistics: performance measures, block bootstrap tests, Cederburg et al. (2020) real-time combination, and
multiple-testing corrections (Holm, Benjamini-Hochberg, Romano-Wolf). Settings come from config.py."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from config import BLOCK, N_BOOT, SEED, CEDERBURG_MIN


def sharpe(z):
    z = np.asarray(z, float); s = z.std(ddof=1)
    return z.mean() / s * np.sqrt(12) if s > 0 else 0.0


def sortino(z):
    z = np.asarray(z, float); dn = np.sqrt(np.mean(np.minimum(z, 0) ** 2))
    return z.mean() * 12 / (dn * np.sqrt(12)) if dn > 0 else np.nan


def maxdd(z, rf):
    wl = np.r_[1.0, np.cumprod(1 + np.asarray(z, float) + np.asarray(rf, float))]
    return (wl / np.maximum.accumulate(wl) - 1).min()


def mt_vt(w, R):
    """Sharpe decomposition SR_managed / SR_unmanaged = MT x VT (gross), with exposures normalised to mean 1."""
    x = w / w.mean(); m = x * R
    return m.mean() / R.mean(), R.std() / m.std()


def perf(z, rf, w=None, R=None):
    out = {"Sharpe": sharpe(z), "Sortino": sortino(z), "ret_%": 1200 * np.mean(z), "vol_%": 100 * np.std(z, ddof=1) * np.sqrt(12),
           "MaxDD_%": 100 * maxdd(z, rf)}
    if w is not None:
        out["avg_w"] = float(np.mean(w))
        if R is not None:
            out["MT"], out["VT"] = mt_vt(w, R)
    return out


def boot_idx(n, rng=None):
    """Circular block bootstrap indices, (N_BOOT, n), blocks of BLOCK months."""
    rng = rng or np.random.default_rng(SEED)
    st = rng.integers(0, n, (N_BOOT, int(np.ceil(n / BLOCK))))
    return ((st[:, :, None] + np.arange(BLOCK)) % n).reshape(N_BOOT, -1)[:, :n]


def sr_rows(M):
    """Sharpe ratio of every row of M (rows = bootstrap samples); 0 for an all-cash sample."""
    sd = M.std(-1, ddof=1)
    return np.where(sd > 0, M.mean(-1) / np.where(sd > 0, sd, 1) * np.sqrt(12), 0.0)


def dsr_test(a, b, I=None):
    """SR(a) - SR(b) with a paired block bootstrap. Returns d, CI low, CI high, two-sided p, and the bootstrap draws."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    I = boot_idx(len(a)) if I is None else I
    d = sharpe(a) - sharpe(b); db = sr_rows(a[I]) - sr_rows(b[I]); se = db.std(ddof=1)
    lo, hi = np.percentile(db - db.mean() + d, [2.5, 97.5])
    return d, lo, hi, (2 * (1 - norm.cdf(abs(d) / se)) if se > 0 else 1.0), db


def fmt(d, lo, hi):
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def cederburg(u, m, min_obs=CEDERBURG_MIN):
    """Cederburg et al. (2020) real-time test: each month, mean-variance weights on (unmanaged, managed) estimated on all
    PAST months (at least min_obs), scaled to the unmanaged past variance (Sharpe unaffected). Returns the combination's
    monthly return and the weight put on the managed series. Re-weighting between the two is not charged extra costs."""
    X = pd.concat([u, m], axis=1).dropna(); vals = X.to_numpy(); out, wm, ix = [], [], []
    for t in range(min_obs, len(X)):
        past = vals[:t]; mu, S = past.mean(0), np.cov(past.T)
        b = np.linalg.solve(S, mu); q = b @ S @ b
        b = b * np.sqrt(S[0, 0] / q) if q > 0 else np.array([1.0, 0.0])
        out.append(vals[t] @ b); wm.append(b[1] / (abs(b[0]) + abs(b[1]))); ix.append(X.index[t])
    return pd.Series(out, index=ix), pd.Series(wm, index=ix)


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); out = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); out[i] = run
    return out


def bh(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); out = np.empty(m); run = 1.0
    for k in range(m - 1, -1, -1):
        i = o[k]; run = min(run, p[i] * m / (k + 1)); out[i] = run
    return out


def romano_wolf(t_obs, t_star):
    """Step-down max-|t| (Romano and Wolf 2005). t_obs: (m,) observed t-stats; t_star: (m, B) centred, studentised
    bootstrap draws from ONE joint bootstrap (same resampled months for every test), so correlation between tests is kept."""
    t_obs = np.abs(np.asarray(t_obs, float)); T = np.abs(np.asarray(t_star, float))
    o = np.argsort(-t_obs); out = np.empty(len(t_obs)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (T[o[k:]].max(0) >= t_obs[i]).mean()); out[i] = run
    return out
