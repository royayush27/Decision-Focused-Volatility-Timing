# archived in old/scipy_garch: run from anywhere, paths below are relative to the project root
import os as _os, sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
for _p in (_ROOT, _P(__file__).resolve().parent):
    _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)

import numpy as np, pandas as pd
from math import erf, sqrt
from scipy.optimize import minimize
from scipy.signal import lfilter
from garch_x import load_inputs
from backtest import simulate, metrics

U = ""
mf = pd.read_csv(U + "results_1970/market_forecasts.csv", index_col="month")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
rv = mf["rv"]
r, x = load_inputs(U + "data/ff_daily_garchdata_long.csv", U + "data/VIXCLS.csv")   # percent, percent^2/day
vix = np.sqrt(252 * x)
months = pd.period_range("1995-01", "2025-12", freq="M"); idx = months.astype(str)

# ---- origin data per month: VIX close on last day before the month, shocks, VIX history
vix_last, sh5, rv_prev = [], [], []
for m in months:
    past = vix.index < m.start_time
    vix_last.append(vix[past].iloc[-1])
    sh5.append((r[past].iloc[-5:] ** 2).sum() / 1e4)
    rv_prev.append(rv[str(m - 1)])
D = pd.DataFrame({"vix": vix_last, "sh5": sh5, "rv_prev": rv_prev}, index=idx)
D["raw"] = 21 * (D.vix ** 2 / 252) / 1e4
rvt = rv.reindex(idx)

def expanding_scale(f, burn):
    k = (rvt / f).expanding().mean().shift(1); k.iloc[:burn] = np.nan
    return f * k

# ---- ARMA(1,1) / AR(1) on daily log VIX, expanding refits, CSS estimation
def css(par, z0):
    m, phi, th = par
    e = lfilter([1, -phi], [1, th], z0 - m)
    return np.sum(e[1:] ** 2)
def arma_forecast(ly, arma=True):
    bnds = [(None, None), (-0.999, 0.9999), (-0.95, 0.95) if arma else (0, 0)]
    res = minimize(css, [ly.mean(), 0.98, 0.0], args=(ly,), method="L-BFGS-B", bounds=bnds)
    m, phi, th = res.x
    e = lfilter([1, -phi], [1, th], ly - m)
    z1 = phi * (ly[-1] - m) + th * e[-1]
    mu = m + z1 * phi ** np.arange(21)               # log VIX path, days 1..21
    return np.exp(2 * mu).sum() / 252 / 1e4, res.x
lv = np.log(vix)
fa, fb = [], []
for m in months:
    ly = lv[lv.index < m.start_time].to_numpy()
    fa.append(arma_forecast(ly, False)[0]); fb.append(arma_forecast(ly, True)[0])
D["AR1_VIX"], D["ARMA11_VIX"] = fa, fb

# ---- direct monthly regressions in logs (expanding OLS), Duan smearing
def reg_forecast(cols, burn=36):
    out = pd.Series(np.nan, index=idx)
    Y = np.log(rvt.to_numpy())
    X = np.column_stack([np.ones(len(idx))] + [np.log(D[c].to_numpy()) for c in cols])
    for i in range(burn, len(idx)):
        b, *_ = np.linalg.lstsq(X[:i], Y[:i], rcond=None)
        resid = Y[:i] - X[:i] @ b
        out.iloc[i] = np.exp(X[i] @ b) * np.mean(np.exp(resid))
    return out
D["REG: VIX"] = reg_forecast(["raw"])
D["REG: VIX + prev-month RV"] = reg_forecast(["raw", "rv_prev"])
D["REG: VIX + prev RV + last-5d shocks"] = reg_forecast(["raw", "rv_prev", "sh5"])
D["REG: prev RV + last-5d shocks (no VIX)"] = reg_forecast(["rv_prev", "sh5"])

gx = pd.read_csv("old/scipy_garch/forecasts_vix/garchx_vix_stat.csv", index_col="month")["v_hat"]
g0 = pd.read_csv("old/scipy_garch/forecasts_vix/garch_1990_stat.csv", index_col="month")["v_hat"]
F = pd.DataFrame(index=idx)
BURN = 36
for c, col in [("VIX2 raw (scaled)", D.raw), ("AR(1) log VIX (scaled)", D.AR1_VIX), ("ARMA(1,1) log VIX (scaled)", D.ARMA11_VIX),
               ("GARCH-X (scaled)", gx), ("GARCH same window (scaled)", g0)]:
    F[c] = expanding_scale(col, BURN)
for c in [c for c in D if c.startswith("REG")]:
    F[c] = D[c]

def qlike(a, f): q = a / f; return q - np.log(q) - 1
def dm(d, lags=3):
    d = np.asarray(d, float); n = len(d); e = d - d.mean(); v = e @ e / n
    for k in range(1, lags + 1): v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    t = d.mean() / np.sqrt(v / n); return t, 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
ref = "VIX2 raw (scaled)"
for lab, (a, b) in {"1998-2024": ("1998-01", "2024-12"), "2000-2024": ("2000-01", "2024-12")}.items():
    R = rvt.loc[a:b]; rows = []
    for c in F:
        f = F[c].loc[a:b]; row = {"forecaster": c, "QLIKE": qlike(R, f).mean(), "bias": f.mean() / R.mean()}
        if c != ref:
            d = qlike(R, F[ref].loc[a:b]) - qlike(R, f); t, p = dm(d); row["dQLIKE (VIX rule minus this)"] = d.mean(); row["DM_p"] = p
        rows.append(row)
    print("== FORECAST ACCURACY", lab, "(positive dQLIKE = this forecaster better than scaled VIX^2)")
    print(pd.DataFrame(rows).set_index("forecaster").round(4).to_string())
D.to_csv("old/scipy_garch/forecasts_vix/rules_features.csv"); F.to_csv("old/scipy_garch/forecasts_vix/rules_forecasts.csv")
last = pd.Series({"AR1": arma_forecast(lv.to_numpy(), False)[1], "ARMA": arma_forecast(lv.to_numpy(), True)[1]})
print("\nfull-sample AR(1) [mean logVIX, phi, theta]:", last["AR1"].round(4), "| ARMA(1,1):", last["ARMA"].round(4))

# ---- threshold rules and continuous rules through the backtest
ex = mf["ret_m"].reindex(idx); rf = rf_all.reindex(idx)
CAL = ("1995-01", "1999-12"); PER = {"dev 2000-2024": ("2000-01", "2024-12"), "check 2025": ("2025-01", "2025-12")}
from calibrate_c import solve_c, exposure
rules = {"Unmanaged": pd.Series(1.0, index=idx)}
for K in (13, 15, 17, 20, 25, 30):
    for lo in (0.0, 0.5):
        rules[f"VIX>{K} -> {lo:g}x"] = pd.Series(np.where(D.vix.to_numpy() > K, lo, 1.0), index=idx)
cx = solve_c(D.raw.loc[CAL[0]:CAL[1]]); rules["MM continuous: VIX^2"] = pd.Series(exposure(D.raw, cx), index=idx)
cg = solve_c(gx.loc[CAL[0]:CAL[1]]); rules["MM continuous: GARCH-X"] = pd.Series(exposure(gx, cg), index=idx)
cg0 = solve_c(g0.loc[CAL[0]:CAL[1]]); rules["MM continuous: GARCH (same window)"] = pd.Series(exposure(g0, cg0), index=idx)
sims = {}
for k, w in rules.items():
    s = simulate(w, ex, rf, 0.0005); s.index = idx; sims[k] = s
for lab, (a, b) in PER.items():
    rows = []
    for k, s in sims.items():
        m_ = metrics(s.loc[a:b]); avg = s.loc[a:b, "w"].mean()
        st = simulate(pd.Series(avg, index=idx[(idx >= a) & (idx <= b)]), ex.loc[a:b], rf.loc[a:b], 0.0005)
        st.index = idx[(idx >= a) & (idx <= b)]; u_static = metrics(st).Utility
        rows.append({"rule": k, "Sharpe": m_.Sharpe, "Utility": m_.Utility, "AvgW": avg, "Turnover": m_.Turnover,
                     "timing gain (U - static U)": m_.Utility - u_static})
    print("\n== BACKTEST", lab, "5 bps, gamma 3")
    print(pd.DataFrame(rows).set_index("rule").round(4).to_string())
