"""Main evaluation, market (Mkt-RF): 1998-2025, with sub-periods 1998-2015, last 10 years (2016-2025), last 5 years (2021-2025).
Question: does volatility timing still work, and do the fixes from exo_indicators/ help?
Everything is real time: c and every estimated parameter come from data before 1998 and are then frozen.
  Calibration: c gives mean exposure 1 on 1970-1997 (rules on RV / GARCH); on 1993-1997 for VIX-based rules
  (VIX starts in 1990; the VRP Kalman parameters are estimated on 1990-1997). w in [0, 2], 5 bps per unit turnover.
Rules:
  Unmanaged                      w = 1
  MM on RV                       w = c / last month's realised variance (Moreira-Muir 2017)
  MM on GARCH                    w = c / GARCH(1,1) forecast of this month's variance (results_1970, walk-forward QMLE)
  MM on VIX^2                    w = c / VIX^2 at the end of last month (illustration: best risk forecast, worst mean timing)
  Xu switches (GARCH)            MM on GARCH; cash if the past 12-month mean return < 0; exposure 1 if the 60-month
                                 risk-return intercept <= 0 (Xu 2024)
  Conditional quintiles (GARCH)  MM on GARCH only when last month's RV is in its top / bottom 20% (expanding), else 1
  Recovery state (RV)            MM on RV, but at least 1 when last month's RV is in its top 20% and 5-day RV < 63-day RV
  VRP Kalman (GARCH)             w = c * max(mu_hat, 0) / GARCH, mu_hat from the precision-weighted Kalman regression on
                                 VRP = VIX^2 - GARCH (exo_indicators/vrp_kalman.py), parameters from 1990-1997
Statistics on monthly net excess returns; MT x VT = SR ratio vs unmanaged (exact on gross returns); paired 6-month block
bootstrap for dSR. Note the precision: with 5 years of data the standard error of a Sharpe ratio is about 0.45."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c

PERIODS = {"1998-2025 (main)": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"),
           "2016-2025 (last 10y)": ("2016-01", "2025-12"), "2021-2025 (last 5y)": ("2021-01", "2025-12")}
CAL, CAL_VIX = ("1970-01", "1997-12"), ("1993-01", "1997-12")

mb = pd.read_csv("data/ff_monthly_baselinedata.csv")
mk = mb[mb["series"] == "Mkt-RF"].set_index("month").sort_index().loc["1963-07":"2025-12"]
mf = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")
rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"].reindex(mk.index)
e, rv1 = mk["ret_m"], mk["rv"].shift(1)
vg = mf["garch_rv_forecast"].reindex(mk.index)
vx = pd.read_csv("data/VIXCLS.csv"); vx.columns = ["date", "vix"]
vx["date"] = pd.to_datetime(vx["date"]); vx["vix"] = pd.to_numeric(vx["vix"], errors="coerce")
vix = vx.dropna().set_index("date")["vix"].resample("ME").last(); vix.index = vix.index.strftime("%Y-%m")
iv = (vix.shift(1) ** 2 / 12 / 1e4).reindex(mk.index)                       # VIX^2 at the end of t-1

d = pd.read_csv("data/ff_daily_garchdata_long.csv"); d = d[d["series"] == "Mkt-RF"].copy()
d["date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y"); r_d = d.set_index("date")["ret"].sort_index()
eom = r_d.groupby(r_d.index.to_period("M")).apply(lambda z: z.index[-1])
rv5 = pd.Series(((r_d ** 2).rolling(5).sum() * 21 / 5).loc[eom.values].to_numpy(), index=eom.index.astype(str)).shift(1).reindex(mk.index)
rv63 = pd.Series(((r_d ** 2).rolling(63).sum() * 21 / 63).loc[eom.values].to_numpy(), index=eom.index.astype(str)).shift(1).reindex(mk.index)


def mm(v, cal=CAL):
    return pd.Series(exposure(v, solve_c(v.loc[cal[0]:cal[1]])), index=v.index)


def rolling_intercept(r, v, window=60):
    a = pd.Series(np.nan, index=r.index)
    for i in range(window, len(r)):
        y, x = r.iloc[i - window:i].to_numpy(), v.iloc[i - window:i].to_numpy(); ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 48:
            a.iloc[i] = np.polyfit(x[ok], y[ok], 1)[1]
    return a


def kf(r, x, vv, kappa, q, p0=1e-4):
    th = np.zeros(2); P = np.eye(2) * p0; Q = np.eye(2) * q; mu = np.empty(len(r)); ll = 0.0
    for t in range(len(r)):
        P = P + Q; h = np.array([1.0, x[t]]); mu[t] = h @ th
        s = h @ P @ h + kappa * vv[t]; k = P @ h / s; err = r[t] - mu[t]
        ll += -0.5 * (np.log(2 * np.pi * s) + err ** 2 / s); th = th + k * err; P = P - np.outer(k, h) @ P
    return mu, ll


W = {"Unmanaged": pd.Series(1.0, index=mk.index), "MM on RV": mm(rv1), "MM on GARCH": mm(vg)}
W["MM on VIX^2"] = mm(iv.loc["1990-02":], CAL_VIX).reindex(mk.index)
hold = e.rolling(12).mean().shift(1) > 0
scale = rolling_intercept(e, rv1) > 0
W["Xu switches (GARCH)"] = W["MM on GARCH"].where(scale, 1.0).where(hold, 0.0)
q20, q80 = rv1.expanding(120).quantile(0.2), rv1.expanding(120).quantile(0.8)
W["Conditional quintiles (GARCH)"] = W["MM on GARCH"].where((rv1 >= q80) | (rv1 <= q20), 1.0)
rec = (rv1 > q80) & (rv5 < rv63)
W["Recovery state (RV)"] = W["MM on RV"].where(~rec, np.maximum(W["MM on RV"], 1.0))
# VRP Kalman, parameters from 1990-1997 only
ix = mk.loc["1990-02":].index
vrp = (iv - vg).reindex(ix); z = (vrp - vrp.loc[:"1997-12"].mean()) / vrp.loc[:"1997-12"].std()
est = slice("1990-02", "1997-12")
_, kap, q = max((kf(e.loc[est].to_numpy(), z.loc[est].to_numpy(), vg.loc[est].to_numpy(), k, q)[1], k, q)
                for k in np.arange(0.5, 4.01, 0.25) for q in np.r_[0.0, 10.0 ** np.arange(-10, -3.4, 0.25)])
mu = pd.Series(kf(e.loc[ix].to_numpy(), z.to_numpy(), vg.loc[ix].to_numpy(), kap, q)[0], index=ix)
w0 = np.maximum(mu, 0) / vg.loc[ix]
W["VRP Kalman (GARCH)"] = (w0 / w0.loc[CAL_VIX[0]:CAL_VIX[1]].mean()).clip(0, 2).reindex(mk.index)
print(f"VRP Kalman parameters (1990-1997): kappa {kap:.2f}, q {q:.1e}")

nets = {}
for k, w in W.items():
    w = w.loc["1990-02":] if k in ("MM on VIX^2", "VRP Kalman (GARCH)") else w.loc["1970-01":]
    sim = simulate(w.fillna(1.0), e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index
    nets[k] = (sim.net_ex, w)

pd.set_option("display.width", 220)
rng = np.random.default_rng(0)
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
for plab, (a, b) in PERIODS.items():
    R = e.loc[a:b]; rows = {}
    for k, (net, w) in nets.items():
        zz = net.loc[a:b]; x = w.loc[a:b]; xn = x / x.mean(); wl = (1 + zz + rf.loc[a:b]).cumprod()
        rows[k] = {"SR_net": zz.mean() / zz.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(), "VT": R.std() / (xn * R).std(),
                   "ann_ret_%": 1200 * zz.mean(), "ann_vol_%": 100 * zz.std() * np.sqrt(12), "MaxDD": (wl / wl.cummax() - 1).min(),
                   "avg_w": x.mean()}
    T = pd.DataFrame(rows).T
    n = len(R); BB, BL = 4000, 6
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    for ref, col in (("Unmanaged", "dSR vs unmanaged [95% CI]"), ("MM on RV", "dSR vs MM on RV [95% CI]")):
        out = []
        for k in T.index:
            if k == ref:
                out.append(""); continue
            A, B = nets[k][0].loc[a:b].to_numpy(), nets[ref][0].loc[a:b].to_numpy()
            d0 = T.loc[k, "SR_net"] - T.loc[ref, "SR_net"]; dd = srb(A[I]) - srb(B[I]); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
            out.append(f"{d0:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        T[col] = out
    print(f"\n======== {plab}: {n} months, unmanaged SR {T.loc['Unmanaged', 'SR_net']:.2f} (SE of a Sharpe ratio about {np.sqrt((1 + 0.5 * T.loc['Unmanaged', 'SR_net'] ** 2) / (n / 12)):.2f})")
    print(T.round(3).to_string())
