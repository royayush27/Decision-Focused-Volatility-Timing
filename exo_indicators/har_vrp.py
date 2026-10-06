"""HAR / HAR+GARCH forecasters and vol + expected-return rules (market).

Part 1  log RV_t forecasts by expanding OLS at month-end origins: GARCH, HAR (day / week / month squared
        returns), HAR+GARCH, VIX, HAR+GARCH+VIX.
Part 2  new EPU categories (trade, monetary, national security; lag t-2) added to the GARCH + lagged-RV baseline.
Part 3  rules w = c * max(mu_hat, 0) / v_hat. mu_hat by expanding OLS on one predictor known at the end of t-1:
        historical mean (= Moreira-Muir), variance risk premium VRP = VIX^2 - RV (Bollerslev-Tauchen-Zhou 2009),
        log EPU, and an equal-weight combination of nine single-predictor forecasts (Rapach-Strauss-Zhou 2010).
VIX era: estimation from 1990, OOS from 1995, c calibrated 1995-2004, evaluation 2005-2024, kappa 5 bps.
"""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd

import exo_test as E
from backtest import simulate
from calibrate_c import exposure, solve_c

fc, X, ind, y, B, rf = E.fc, E.X, E.ind, E.y, E.B, E.rf
ex, rv = fc["ret_m"], fc["rv"]
EST, OOS, CAL, EV = "1990-02", "1995-01", ("1995-01", "2004-12"), ("2005-01", "2024-12")

# ---------- HAR components at each month-end origin (last trading day of t-1)
d = pd.read_csv("data/ff_daily_garchdata_long.csv")
d = d[d.series == "Mkt-RF"].copy()
d["Date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
d = d.sort_values("Date")
r2 = d["ret"].to_numpy() ** 2
cum = np.r_[0.0, np.cumsum(r2)]
last_idx = d.groupby("month")["Date"].idxmax()
pos = {ix: k for k, ix in enumerate(d.index)}
H = pd.DataFrame(index=fc.index, columns=["rv_d", "rv_w", "rv_m"], dtype=float)
months = list(last_idx.index)
for j, mth in enumerate(months[:-1]):
    nxt = months[j + 1]
    if nxt not in H.index:
        continue
    k = pos[last_idx[mth]] + 1                     # number of days up to and including the origin
    H.loc[nxt] = [22 * r2[k - 1], 22 / 5 * (cum[k] - cum[k - 5]), cum[k] - cum[k - 22]]
H["rv_d"] = np.maximum(H["rv_d"], 0.05 * H["rv_m"])  # floor: a zero return would give log(0)
X["hd"], X["hw"], X["hm"] = np.log(H["rv_d"]), np.log(H["rv_w"]), np.log(H["rv_m"])
X["lg"] = B["lg"]
X["VRP"] = (E.vix.shift(1).reindex(fc.index) - fc["rv_lag1"])      # VIX^2 (end of t-1) minus RV of t-1
for c, s in (("lTRADE", "EPU_TRADE"), ("lMON", "EPU_MON"), ("lNATSEC", "EPU_NATSEC")):
    X[c] = np.log(ind[s].shift(2).reindex(fc.index))


def ols_oos(target, cols, est=EST, oos=OOS):
    Z = X[cols].copy(); Z.insert(0, "c", 1.0)
    f = pd.Series(np.nan, index=fc.index); s2 = pd.Series(np.nan, index=fc.index)
    for t in fc.loc[oos:"2025-12"].index:
        tr = Z.loc[est:t].iloc[:-1]; yt = target.loc[tr.index]
        ok = tr.notna().all(axis=1) & yt.notna()
        if ok.sum() < 48 or Z.loc[t].isna().any():
            continue
        A, b = tr[ok].to_numpy(), yt[ok].to_numpy()
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        f[t] = Z.loc[t].to_numpy() @ beta; s2[t] = (b - A @ beta).var(ddof=A.shape[1])
    return f, s2


sr = lambda z: z.mean() / z.std(ddof=1) * np.sqrt(12)


def backtest(w):
    s = simulate(w.fillna(1.0), ex, rf, 5e-4); s.index = fc.index
    return s.net_ex.loc[EV[0]:EV[1]]


def boot(z1, z0, BB=4000, BL=6, seed=0):
    rng = np.random.default_rng(seed); a, b = z1.to_numpy(), z0.to_numpy(); n = len(a)
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    d = (a[I].mean(1) / a[I].std(1, ddof=1) - b[I].mean(1) / b[I].std(1, ddof=1)) * np.sqrt(12); d0 = sr(z1) - sr(z0)
    return d0, np.percentile(d - d.mean() + d0, 2.5), np.percentile(d - d.mean() + d0, 97.5)


# ---------- Part 1: forecasters
models = {"GARCH": ["lg"], "HAR": ["hd", "hw", "hm"], "HAR+GARCH": ["hd", "hw", "hm", "lg"],
          "VIX": ["lVIX2"], "HAR+GARCH+VIX": ["hd", "hw", "hm", "lg", "lVIX2"]}
V, rows = {}, []
a, b = EV
yy, Q, R = y.loc[a:b], rv.loc[a:b], ex.loc[a:b]
ref = None
for k, cols in models.items():
    f, s2 = ols_oos(y, cols)
    v = np.exp(f + s2 / 2); V[k] = v
    w = pd.Series(exposure(v, solve_c(v.loc[CAL[0]:CAL[1]].dropna())), index=fc.index)
    z = backtest(w); W = w.loc[a:b]; x = W / W.mean()
    e = (yy - f.loc[a:b])
    if ref is None:
        ref = e
    q = Q / v.loc[a:b]
    rows.append({"forecaster": k, "R2_logRV_vs_GARCH": 1 - (e ** 2).sum() / (ref ** 2).sum(),
                 "QLIKE": (q - np.log(q) - 1).mean(), "VR": (Q * x ** 2).mean() / Q.mean(),
                 "MT": (x * R).mean() / R.mean(), "SR_net": sr(z), "turnover": w.diff().abs().loc[a:b].mean()})
print(f"Part 1: forecasters, eval {EV}, unmanaged SR {sr(R):.3f}")
print(pd.DataFrame(rows).set_index("forecaster").round(3).to_string())

# ---------- Part 2: EPU categories on the vol side
if __name__ == "__main__":
    E.evaluate("1986-03", "1995-01", ("2005-01", "2024-12"), ("1995-01", "2004-12"),
               ["lTRADE", "lMON", "lNATSEC"], "Part 2: EPU categories (lag t-2) on the vol side")

# ---------- Part 3: vol + expected return
preds = ["DEF", "CS", "TERM", "lEPU", "NS", "VRP", "lTRADE", "lMON", "lNATSEC"]
hist = pd.Series(np.nan, index=fc.index)
for t in fc.loc[OOS:"2025-12"].index:
    hist[t] = ex.loc[EST:t].iloc[:-1].mean()
MU = {"hist (MM)": hist}
for p in preds:
    MU[p] = ols_oos(ex, [p])[0]
MU["combo9"] = pd.concat([MU[p] for p in preds], axis=1).mean(axis=1)
r2 = {k: 1 - ((R - m.loc[a:b]) ** 2).sum() / ((R - hist.loc[a:b]) ** 2).sum() for k, m in MU.items()}
print("\nPart 3a: OOS R2 for monthly market excess returns vs historical mean (Campbell-Thompson), eval", EV)
print("  " + "  ".join(f"{k}: {v:+.4f}" for k, v in r2.items()))


def rule(mu, v):
    sig = np.maximum(mu, 0) / v; sig = sig.where(sig > 0, 1e-12)
    return pd.Series(np.clip(solve_c(1 / sig.loc[CAL[0]:CAL[1]].dropna()) * sig, 0, 2), index=fc.index)


print("\nPart 3b: w = c*max(mu,0)/v. dSR vs the same v with historical mean (95% block-bootstrap CI)")
out = []
for vk in ("GARCH", "HAR+GARCH", "VIX"):
    z0 = backtest(rule(hist, V[vk]))
    for mk in ("VRP", "lEPU", "combo9"):
        w = rule(MU[mk], V[vk]); z = backtest(w); W = w.loc[a:b]; x = W / W.mean()
        d0, lo, hi = boot(z, z0)
        out.append({"v_hat": vk, "mu_hat": mk, "SR_net": sr(z), "SR_MM": sr(z0), "dSR": d0, "lo": lo, "hi": hi,
                    "MT": (x * R).mean() / R.mean(), "VR": (Q * x ** 2).mean() / Q.mean(), "zero_w": (W < 1e-6).mean()})
print(pd.DataFrame(out).round(3).to_string(index=False))
