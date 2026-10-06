import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import solve_c
from risk_dfl import FACTORS

# Exposure rule w = clip(c * v_hat^(-p), 0, 2), v_hat = QMLE GARCH forecast (annual refits, from panel/).
# p = 1 is Moreira-Muir, p = 0 is unmanaged. p is the "curvature" that absorbs the risk-return relation.
# Learn p by decision loss (mean net Sharpe over training months), either per series or pooled across all
# 16 series (one shared p). Annual expanding refits; c re-solved each refit so mean training exposure = 1.

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
GRID = np.round(np.arange(0.0, 1.51, 0.05), 2)
KAPPA, W_MAX = 5e-4, 2.0

m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
data = {}
for s in SERIES:
    ms = m[m["series"] == s].set_index("month").sort_index().loc["1970-01":"2025-12"]
    rf = rf_all.reindex(ms.index)
    ex = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    v = pd.read_csv(f"dfl_diagnosis/panel/{s}_garch.csv", index_col="month")["qmle"]
    data[s] = (v.to_numpy(), ex, rf)
idx = data["Mkt-RF"][1].index
months = pd.PeriodIndex(idx, freq="M")


def rule(v, p, tr):
    z = v ** (-p)
    f = lambda c: np.clip(c * z[tr], 0, W_MAX).mean() - 1.0
    lo, hi = 1e-12, 1.0
    while f(hi) < 0:
        hi *= 4
    from scipy.optimize import brentq
    c = brentq(f, lo, hi, xtol=1e-14)
    return np.clip(c * z, 0, W_MAX)


def sharpe(x):
    return x.mean() / x.std(ddof=1) * np.sqrt(12)


def net(w, ex, rf):
    s = simulate(pd.Series(w, index=ex.index), ex, rf, KAPPA)
    return s.net_ex.to_numpy()


W = {k: {s: np.full(len(idx), np.nan) for s in SERIES} for k in ("unmanaged", "MM p=1", "per-series p", "pooled p")}
chosen = []
for y in range(2000, 2026):
    tr = months < pd.Period(f"{y}-01")
    te = (months >= pd.Period(f"{y}-01")) & (months <= pd.Period(f"{y}-12"))
    score = {s: {} for s in SERIES}
    for s in SERIES:
        v, ex, rf = data[s]
        for p in GRID:
            w = rule(v, p, tr)
            score[s][p] = sharpe(net(w, ex, rf)[tr])
    pooled = max(GRID, key=lambda p: np.mean([score[s][p] for s in SERIES]))
    for s in SERIES:
        v, ex, rf = data[s]
        own = max(GRID, key=lambda p: score[s][p])
        for k, p in (("unmanaged", 0.0), ("MM p=1", 1.0), ("per-series p", own), ("pooled p", pooled)):
            W[k][s][te] = rule(v, p, tr)[te]
        chosen.append({"year": y, "series": s, "own_p": own, "pooled_p": pooled})
    print(y, "pooled p =", pooled, flush=True)

pd.DataFrame(chosen).to_csv("dfl_diagnosis/pooled_p_choices.csv", index=False)
dev = (months >= pd.Period("2000-01")) & (months <= pd.Period("2024-12"))
res, nets = [], {}
for k in W:
    for s in SERIES:
        v, ex, rf = data[s]
        w = W[k][s].copy()
        w[~(months >= pd.Period("2000-01"))] = 1.0       # placeholder before OOS; only dev months are scored
        z = net(w, ex, rf)[dev]
        nets[(k, s)] = z
        res.append({"rule": k, "series": s, "SR": sharpe(z), "avgW": np.nanmean(W[k][s][dev])})
R = pd.DataFrame(res)
P = R.pivot(index="series", columns="rule", values="SR")
print("\n== OOS 2000-2024 net Sharpe by series")
print(P.round(3).to_string())
print("\nmean SR:", P.mean().round(4).to_dict())

rng = np.random.default_rng(0)
n = dev.sum(); B, BL = 2000, 6
st = rng.integers(0, n, (B, int(np.ceil(n / BL))))
I = ((st[:, :, None] + np.arange(BL)) % n).reshape(B, -1)[:, :n]
def boot(A, Bn):
    d = np.zeros(B); d0 = []
    for s in SERIES:
        a, b = nets[(A, s)], nets[(Bn, s)]
        sa = a[I].mean(1) / a[I].std(1, ddof=1); sb = b[I].mean(1) / b[I].std(1, ddof=1)
        d += (sa - sb) * np.sqrt(12); d0.append(sharpe(a) - sharpe(b))
    d /= len(SERIES); m0 = np.mean(d0)
    return {"A": A, "B": Bn, "mean dSR": m0, "CI lo": np.percentile(d - d.mean() + m0, 2.5),
            "CI hi": np.percentile(d - d.mean() + m0, 97.5), "p": np.mean(np.abs(d - d.mean()) >= abs(m0)),
            "series A better": int(np.sum(np.array(d0) > 0))}
T = pd.DataFrame([boot("pooled p", "MM p=1"), boot("pooled p", "unmanaged"), boot("MM p=1", "unmanaged"),
                  boot("per-series p", "MM p=1"), boot("pooled p", "per-series p")])
print("\n== panel block bootstrap, common blocks across series")
print(T.round(4).to_string(index=False))
