"""'When to trust what', done formally: a Kalman filter for a time-varying expected return.
Model (monthly): r_t = mu_t + e_t, Var(e_t) = v_t (last month's realised variance, known at the end of t-1);
                 mu_t = mu_{t-1} + u_t,  Var(u_t) = q.
The filter trusts each new return in proportion to its precision: gain K_t = P_t / (P_t + v_t). A crash month (large v_t)
moves the estimate of mu very little; a calm month moves it more. q (how fast the expected return can change) is estimated
by maximum likelihood on 1965-1999 only and then fixed. Two versions:
  naive: Var(e_t) = v_t exactly (lagged RV taken at face value);
  full:  Var(e_t) = kappa * v_t with kappa estimated jointly with q (lagged RV understates next month's return variance). Rule: w = c * max(mu_hat_{t|t-1}, 0) / v_t, c calibrated 1975-1999
(mean exposure 1, w in [0, 2]); benchmark = same rule with the expanding historical mean (= Moreira-Muir up to c).
Evaluation 2000-2024, 5 bps; market and 16 series (each with its own q)."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis"):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import solve_c
from risk_dfl import FACTORS

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
EST, CAL, EV = ("1965-01", "1999-12"), ("1975-01", "1999-12"), ("2000-01", "2024-12")
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]


def kalman(r, v, q, mu0, p0):
    """One-step-ahead predictions mu_{t|t-1}, their variance P_{t|t-1}, and the log-likelihood."""
    n = len(r); mu_pred, p_pred = np.empty(n), np.empty(n); mu, p, ll = mu0, p0, 0.0
    for t in range(n):
        p = p + q; mu_pred[t], p_pred[t] = mu, p
        s = p + v[t]; k = p / s
        ll += -0.5 * (np.log(2 * np.pi * s) + (r[t] - mu) ** 2 / s)
        mu, p = mu + k * (r[t] - mu), (1 - k) * p
    return mu_pred, p_pred, ll


def rule(mu, v):
    sig = (np.maximum(mu, 0) / v).where(lambda z: z > 0, 1e-12)
    return pd.Series(np.clip(solve_c(1 / sig.loc[CAL[0]:CAL[1]].dropna()) * sig, 0, 2), index=v.index)


rows, nets = [], {}
GRID = 10.0 ** np.arange(-9, -2.9, 0.25)
KAPPA = np.arange(0.5, 4.01, 0.25)
for s in SERIES:
    ms = m[m["series"] == s].set_index("month").sort_index().loc["1963-07":"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    v = ms["rv"].shift(1)
    e, v, rf = e.loc["1965-01":], v.loc["1965-01":], rf.loc["1965-01":]
    est = e.loc[EST[0]:EST[1]].to_numpy(); vest = v.loc[EST[0]:EST[1]].to_numpy()
    mu0, p0 = 0.0, 0.01 ** 2                                   # vague prior: mean 0, sd 1% per month
    hist = e.expanding().mean().shift(1)
    R = e.loc[EV[0]:EV[1]]
    fits = {"Kalman naive": (1.0, GRID[int(np.argmax([kalman(est, vest, q, mu0, p0)[2] for q in GRID]))])}
    best = max(((kalman(est, k * vest, q, mu0, p0)[2], k, q) for k in KAPPA for q in GRID))
    fits["Kalman full"] = (best[1], best[2])
    mus = {"hist mean (MM)": (hist, np.nan, np.nan, np.nan, np.nan)}
    for lab, (kap, q) in fits.items():
        mu_pred, p_pred, _ = kalman(e.to_numpy(), kap * v.to_numpy(), q, mu0, p0)
        pp = pd.Series(p_pred, index=e.index).loc[EV[0]:EV[1]]; vv = kap * v.loc[EV[0]:EV[1]]; g = pp / (pp + vv)
        mus[lab] = (pd.Series(mu_pred, index=e.index), kap, q, g[vv <= vv.quantile(0.5)].mean(), g[vv >= vv.quantile(0.9)].mean())
    for lab, (mu, kap, q, k_calm, k_vol) in mus.items():
        r2 = 1 - ((R - mu.loc[R.index]) ** 2).sum() / ((R - hist.loc[R.index]) ** 2).sum()
        w = rule(mu, v); sim = simulate(w, e, rf, 5e-4); sim.index = e.index
        z = sim.net_ex.loc[EV[0]:EV[1]]; nets[(lab, s)] = z.to_numpy()
        x = w.loc[EV[0]:EV[1]]; xn = x / x.mean()
        rows.append({"series": s, "mu": lab, "SR": z.mean() / z.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(),
                     "VT": R.std() / (xn * R).std(), "kappa": kap, "q_sd_%": 100 * np.sqrt(q),
                     "gain_calm": k_calm, "gain_top10%_vol": k_vol, "R2oos_%": 100 * r2})
T = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("Market 2000-2024 (q = variance of the monthly change in mu; gain = weight on the newest return)")
print(T[T.series == "Mkt-RF"].set_index("mu")[["SR", "MT", "VT", "kappa", "q_sd_%", "gain_calm", "gain_top10%_vol", "R2oos_%"]].round(4).to_string())
for lab in ("Kalman naive", "Kalman full"):
    K = T[T.mu == lab].set_index("series")
    print(f"\n{lab} per series")
    print(K[["kappa", "q_sd_%", "gain_calm", "gain_top10%_vol", "R2oos_%"]].round(4).T.to_string())
n = len(nets[("Kalman full", "Mkt-RF")]); BB, BL = 4000, 6; rng = np.random.default_rng(0)
st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
P = T.pivot(index="series", columns="mu", values="SR")
print()
for lab in ("Kalman naive", "Kalman full"):
    for slab, sers in (("market", ["Mkt-RF"]), ("panel 16", SERIES)):
        dd = np.mean([srb(nets[(lab, s)][I]) - srb(nets[("hist mean (MM)", s)][I]) for s in sers], axis=0)
        d0 = (P.loc[sers, lab] - P.loc[sers, "hist mean (MM)"]).mean(); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
        print(f"  {lab:12s} - MM, {slab:8s}: dSR {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]  better {int((P.loc[sers, lab] > P.loc[sers, 'hist mean (MM)']).sum())}/{len(sers)}")
