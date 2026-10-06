"""'When to trust the VRP signal vs plain vol timing', with a Kalman filter on the predictive regression itself.
Model (monthly, market): r_t = a_t + b_t * z_{t-1} + e_t,  Var(e_t) = kappa * v_t,
  z = ex-ante variance risk premium VRP = VIX^2 (end of t-1) - GARCH forecast for month t, standardised with 1990-2004 statistics,
  v_t = GARCH forecast; (a_t, b_t) follow random walks with variance q each.
If the VRP relation is stable, q -> 0 and the filter is recursive least squares (the static VRP rule tested in the session);
if it switches on and off, q > 0 and b_t tracks when VRP has been predicting well. The filter's own uncertainty P_t says how
much to trust it: mu_hat = a_t + b_t z_{t-1}. kappa and q by maximum likelihood on 1990-02..2004-12, then fixed.
Rule: w = c * max(mu_hat, 0) / v_t, c calibrated 2000-2004 (mean exposure 1), w in [0, 2], evaluation 2005-2024, 5 bps.
Benchmarks: historical mean (= Moreira-Muir with GARCH up to c) and the static expanding-OLS VRP rule.
With q = 0 the filter is recursive weighted least squares (weights 1 / v_t) with a prior shrinking b towards 0.
Robustness: prior sd of (a, b) 0.5% to 4%; panel of 16 series with market VRP, v = last month's RV, kappa 1, q 0."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
sys.path.insert(0, str(_ROOT / "dfl_diagnosis"))
from risk_dfl import FACTORS

mf = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")
rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
vx = pd.read_csv("data/VIXCLS.csv"); vx.columns = ["date", "vix"]
vx["date"] = pd.to_datetime(vx["date"]); vx["vix"] = pd.to_numeric(vx["vix"], errors="coerce")
vix = vx.dropna().set_index("date")["vix"].resample("ME").last(); vix.index = vix.index.strftime("%Y-%m")
idx = mf.loc["1990-02":"2025-12"].index
e, rf = mf["ret_m"].reindex(idx), rf.reindex(idx)
v = mf["garch_rv_forecast"].reindex(idx)                               # forecast for month t, made at the end of t-1
vrp = (vix.reindex(mf.index).shift(1) ** 2 / 12 / 1e4).reindex(idx) - v  # both known at the end of t-1
EST, CAL, EV = ("1990-02", "2004-12"), ("2000-01", "2004-12"), ("2005-01", "2024-12")
z = (vrp - vrp.loc[EST[0]:EST[1]].mean()) / vrp.loc[EST[0]:EST[1]].std()


def kf(r, x, vv, kappa, q, beta0=(0.0, 0.0), p0=1e-4):
    n = len(r); th = np.array(beta0, float); P = np.eye(2) * p0; Q = np.eye(2) * q
    mu, b, bsd, ll = np.empty(n), np.empty(n), np.empty(n), 0.0
    for t in range(n):
        P = P + Q; h = np.array([1.0, x[t]])
        mu[t] = h @ th; b[t] = th[1]; bsd[t] = np.sqrt(P[1, 1])
        s = h @ P @ h + kappa * vv[t]; k = P @ h / s; err = r[t] - mu[t]
        ll += -0.5 * (np.log(2 * np.pi * s) + err ** 2 / s)
        th = th + k * err; P = P - np.outer(k, h) @ P
    return mu, b, bsd, ll


est = slice(EST[0], EST[1])
best = max((kf(e.loc[est].to_numpy(), z.loc[est].to_numpy(), v.loc[est].to_numpy(), k, q)[3], k, q)
           for k in np.arange(0.5, 4.01, 0.25) for q in np.r_[0.0, 10.0 ** np.arange(-10, -3.4, 0.25)])
_, kap, q = best
mu_k, b_k, bsd_k, _ = kf(e.to_numpy(), z.to_numpy(), v.to_numpy(), kap, q)
mu_k, b_k, bsd_k = (pd.Series(a, index=idx) for a in (mu_k, b_k, bsd_k))

hist, ols = pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
for i, t in enumerate(idx):
    if t < CAL[0]:
        continue
    tr = idx[:i]; hist[t] = e.loc[tr].mean()
    bb, aa = np.polyfit(z.loc[tr], e.loc[tr], 1); ols[t] = aa + bb * z.loc[t]


def rule(mu):
    w0 = np.maximum(mu, 0) / v
    return (w0 / w0.loc[CAL[0]:CAL[1]].mean()).clip(0, 2)


rows, nets = {}, {}
R = e.loc[EV[0]:EV[1]]
for lab, mu in (("historical mean (MM, GARCH)", hist), ("VRP, static OLS", ols), ("VRP, Kalman (time-varying)", mu_k)):
    w = rule(mu).loc[CAL[0]:]
    sim = simulate(w, e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index
    zz = sim.net_ex.loc[EV[0]:EV[1]]; nets[lab] = zz.to_numpy()
    x = w.loc[EV[0]:EV[1]]; xn = x / x.mean()
    r2 = 1 - ((R - mu.loc[R.index]) ** 2).sum() / ((R - hist.loc[R.index]) ** 2).sum()
    rows[lab] = {"SR_net": zz.mean() / zz.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(), "VT": R.std() / (xn * R).std(),
                 "OOS_R2_%": 100 * r2, "months_in_cash": int((x < 1e-6).sum()), "turnover": x.diff().abs().mean()}
pd.set_option("display.width", 200)
print(f"MLE on {EST[0]}..{EST[1]}: kappa {kap:.2f}, q {q:.2e} (sd of the monthly change in b: {np.sqrt(q):.2e} per 1-sd move in VRP)")
print(f"\nMarket, evaluation {EV[0]} to {EV[1]}, net of 5 bps")
print(pd.DataFrame(rows).T.round(3).to_string())
bt = (b_k / bsd_k).loc[EV[0]:EV[1]]
print(f"\nHow much the filter trusts VRP: b_t / sd(b_t) above 2 in {100 * (bt > 2).mean():.0f}% of months, below -2 in {100 * (bt < -2).mean():.0f}%")
print("b_t (monthly return per 1-sd VRP, %) at selected dates:",
      ", ".join(f"{t} {100 * b_k[t]:+.2f} (sd {100 * bsd_k[t]:.2f})" for t in ("2005-01", "2008-12", "2012-12", "2016-12", "2020-12", "2024-12")))
n = len(R); BB, BL = 4000, 6; rng = np.random.default_rng(0)
st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
base = "historical mean (MM, GARCH)"
for lab in list(rows)[1:]:
    a, b = nets[lab], nets[base]; d0 = rows[lab]["SR_net"] - rows[base]["SR_net"]
    dd = srb(a[I]) - srb(b[I]); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
    print(f"  {lab:28s} - MM: dSR {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")

print("\nRobustness: prior sd of (a, b)")
for p0 in (0.25e-4, 1e-4, 4e-4, 16e-4):
    mu = pd.Series(kf(e.to_numpy(), z.to_numpy(), v.to_numpy(), kap, q, p0=p0)[0], index=idx)
    w = rule(mu).loc[CAL[0]:]; sim = simulate(w, e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index
    zz = sim.net_ex.loc[EV[0]:EV[1]]; r2 = 1 - ((R - mu[R.index]) ** 2).sum() / ((R - hist[R.index]) ** 2).sum()
    print(f"  prior sd {100 * np.sqrt(p0):.1f}%: SR {zz.mean() / zz.std() * np.sqrt(12):.3f}, OOS R2 {100 * r2:+.2f}%")

SER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
mm = pd.read_csv("data/ff_monthly_baselinedata.csv"); rfa = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
res = {k: [] for k in ("hist", "ols", "kalman")}; NET = {k: [] for k in res}
for s in SER:
    ms = mm[mm["series"] == s].set_index("month").sort_index()
    rr = rfa.reindex(idx); es = (ms["ret_m"] if s in FACTORS else ms["ret_m"] - rfa.reindex(ms.index)).reindex(idx)
    vs = ms["rv"].shift(1).reindex(idx)
    mu_w = pd.Series(kf(es.to_numpy(), z.to_numpy(), vs.to_numpy(), 1.0, 0.0)[0], index=idx)
    h, o = pd.Series(np.nan, index=idx), pd.Series(np.nan, index=idx)
    for i, t in enumerate(idx):
        if t < CAL[0]:
            continue
        tr = idx[:i]; h[t] = es.loc[tr].mean(); bb, aa = np.polyfit(z.loc[tr], es.loc[tr], 1); o[t] = aa + bb * z.loc[t]
    for k, mu in (("hist", h), ("ols", o), ("kalman", mu_w)):
        w0 = np.maximum(mu, 0) / vs; w = (w0 / w0.loc[CAL[0]:CAL[1]].mean()).clip(0, 2).loc[CAL[0]:]
        sim = simulate(w, es.loc[w.index], rr.loc[w.index], 5e-4); sim.index = w.index; zz = sim.net_ex.loc[EV[0]:EV[1]]
        res[k].append(zz.mean() / zz.std() * np.sqrt(12)); NET[k].append(zz.to_numpy())
print("\nPanel, 16 series, market VRP, v = last month's RV")
for k in ("ols", "kalman"):
    dd = np.mean([srb(NET[k][i][I]) - srb(NET["hist"][i][I]) for i in range(16)], axis=0)
    d0 = np.mean(np.array(res[k]) - np.array(res["hist"])); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
    print(f"  VRP {k:6s} - hist mean: mean dSR {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]  better {sum(a > b for a, b in zip(res[k], res['hist']))}/16")
