"""Full decomposition of VIX^2 = expected variance + variance risk premium, used on both sides of the rule:
  denominator = GARCH-X (GARCH with VIX^2, the best risk forecast, from old/scipy_garch/forecasts_vix/garchx_vix_stat.csv)
  numerator   = mu_hat from the precision-weighted Kalman regression on VRP_x = VIX^2 (end of t-1) - GARCH-X forecast.
w = c * max(mu_hat, 0) / v_hat, c calibrated 2000-2004, w in [0, 2], evaluation 2005-2024, 5 bps, market.
Kalman as in vrp_kalman.py (kappa and q by maximum likelihood, here on 1995-2004 because GARCH-X starts in 1995).
Compared with: vol timing on GARCH and on GARCH-X (historical mean in the numerator), and VRP-Kalman on GARCH (vrp_kalman.py)."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate

mf = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
vx = pd.read_csv("data/VIXCLS.csv"); vx.columns = ["date", "vix"]
vx["date"] = pd.to_datetime(vx["date"]); vx["vix"] = pd.to_numeric(vx["vix"], errors="coerce")
vix = vx.dropna().set_index("date")["vix"].resample("ME").last(); vix.index = vix.index.strftime("%Y-%m")
gx = pd.read_csv("old/scipy_garch/forecasts_vix/garchx_vix_stat.csv", index_col="month")["v_hat"]
CAL, EV = ("2000-01", "2004-12"), ("2005-01", "2024-12")
iv = (vix.reindex(mf.index).shift(1) ** 2 / 12 / 1e4)                 # VIX^2 at the end of t-1, monthly variance


def kf(r, x, vv, kappa, q, p0=1e-4):
    th = np.zeros(2); P = np.eye(2) * p0; Q = np.eye(2) * q; mu = np.empty(len(r)); ll = 0.0
    for t in range(len(r)):
        P = P + Q; h = np.array([1.0, x[t]]); mu[t] = h @ th
        s = h @ P @ h + kappa * vv[t]; k = P @ h / s; err = r[t] - mu[t]
        ll += -0.5 * (np.log(2 * np.pi * s) + err ** 2 / s); th = th + k * err; P = P - np.outer(k, h) @ P
    return mu, ll


def vrp_mu(v, start, est_end):
    idx = mf.loc[start:"2025-12"].index; e = mf["ret_m"].reindex(idx); vv = v.reindex(idx)
    vrp = iv.reindex(idx) - vv; z = (vrp - vrp.loc[:est_end].mean()) / vrp.loc[:est_end].std()
    est = slice(start, est_end)
    _, kap, q = max((kf(e.loc[est].to_numpy(), z.loc[est].to_numpy(), vv.loc[est].to_numpy(), k, q)[1], k, q)
                    for k in np.arange(0.5, 4.01, 0.25) for q in np.r_[0.0, 10.0 ** np.arange(-10, -3.4, 0.25)])
    return pd.Series(kf(e.to_numpy(), z.to_numpy(), vv.to_numpy(), kap, q)[0], index=idx), kap, q


hist = mf["ret_m"].loc["1990-02":].expanding().mean().shift(1).reindex(mf.index)
vg = mf["garch_rv_forecast"]
mu_g, kg, qg = vrp_mu(vg, "1990-02", "2004-12")
mu_x, kx, qx = vrp_mu(gx, "1995-01", "2004-12")
print(f"MLE: VRP on GARCH kappa {kg:.2f} q {qg:.1e} (1990-2004) | VRP_x on GARCH-X kappa {kx:.2f} q {qx:.1e} (1995-2004)")
print(f"mean VRP 2005-24, % monthly variance: VIX^2 - GARCH {1e2 * (iv - vg).loc[EV[0]:EV[1]].mean():.3f}, "
      f"VIX^2 - GARCH-X {1e2 * (iv - gx).loc[EV[0]:EV[1]].mean():.3f}; corr of the two VRPs {(iv - vg).corr(iv - gx):.2f}")

e, rf = mf["ret_m"], rf_all.reindex(mf.index)
RULES = {"vol timing, GARCH (MM)": (hist, vg), "vol timing, GARCH-X": (hist, gx),
         "VRP Kalman / GARCH": (mu_g, vg), "VRP_x Kalman / GARCH-X (full decomposition)": (mu_x, gx)}
rows, nets = {}, {}
R = e.loc[EV[0]:EV[1]]
for lab, (mu, v) in RULES.items():
    w0 = (np.maximum(mu, 0) / v).loc[CAL[0]:]; w = (w0 / w0.loc[CAL[0]:CAL[1]].mean()).clip(0, 2)
    sim = simulate(w, e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index
    z = sim.net_ex.loc[EV[0]:EV[1]]; nets[lab] = z.to_numpy(); x = w.loc[EV[0]:EV[1]]; xn = x / x.mean(); wl = (1 + z).cumprod()
    rows[lab] = {"SR_net": z.mean() / z.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(), "VT": R.std() / (xn * R).std(),
                 "MaxDD": (wl / wl.cummax() - 1).min(), "months_in_cash": int((x < 1e-6).sum()), "turnover": x.diff().abs().mean()}
pd.set_option("display.width", 200)
print(f"\nMarket, {EV[0]} to {EV[1]}, net of 5 bps; unmanaged SR {R.mean() / R.std() * np.sqrt(12):.3f}")
print(pd.DataFrame(rows).T.round(3).to_string())
n = len(R); BB, BL = 4000, 6; rng = np.random.default_rng(0)
st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
for a, b in (("VRP_x Kalman / GARCH-X (full decomposition)", "vol timing, GARCH (MM)"),
             ("VRP_x Kalman / GARCH-X (full decomposition)", "VRP Kalman / GARCH"),
             ("vol timing, GARCH-X", "vol timing, GARCH (MM)")):
    d0 = rows[a]["SR_net"] - rows[b]["SR_net"]; dd = srb(nets[a][I]) - srb(nets[b][I]); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
    print(f"  {a} - {b}: dSR {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")
