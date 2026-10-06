import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
DEV = ("2000-01", "2024-12")
KAPPA, GAMMA = 5e-4, 3.0
B, BLOCK = 2000, 6
P = Path("dfl_diagnosis/panel")

m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]


def qlike(a, f):
    q = a / f
    return q - np.log(q) - 1


rows, store = [], {}
for s in SERIES:
    ms = m[m["series"] == s].set_index("month").sort_index().loc["1970-01":"2025-12"]
    rf = rf_all.reindex(ms.index)
    ex = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    rv = ms["rv"]
    fc = {"lagged_rv": ms["rv_lag1"]}
    for mod in ("garch", "gjr"):
        f = P / f"{s}_{mod}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, index_col="month")
        for k in d:
            fc[f"{mod}_{k}"] = d[k]
    a, b = DEV
    R, Q = ex.loc[a:b], rv.loc[a:b]
    un = simulate(pd.Series(1.0, index=ex.index), ex, rf, KAPPA); un.index = ex.index
    for name, v in fc.items():
        w = pd.Series(exposure(v, solve_c(v.loc[:"1999-12"])), index=v.index)
        sim = simulate(w, ex, rf, KAPPA); sim.index = ex.index
        W = w.loc[a:b]; x = W / W.mean()
        net = sim.net_ex.loc[a:b]
        st = simulate(pd.Series(W.mean(), index=R.index), R, rf.loc[a:b], KAPPA)
        u = lambda z: z.mean() * 12 - GAMMA / 2 * z.var(ddof=1) * 12
        wealth = (1 + net + rf.loc[a:b]).cumprod()
        rows.append({"series": s, "forecaster": name, "QLIKE": qlike(Q, v.loc[a:b]).mean(),
                     "VR_rv": (Q * x ** 2).mean() / Q.mean(), "VT": R.std() / (x * R).std(),
                     "MT": (x * R).mean() / R.mean(), "MT_alpha_pa": ((x * R).mean() - R.mean()) * 12, "SR_un": R.mean() / R.std() * np.sqrt(12),
                     "SR_net": net.mean() / net.std() * np.sqrt(12),
                     "timing_gain": u(net) - u(st.net_ex), "AvgW": W.mean(),
                     "MaxDD": (wealth / np.maximum.accumulate(wealth) - 1).min(),
                     "MaxDD_un": ((1 + un.net_ex.loc[a:b] + rf.loc[a:b]).cumprod() /
                                  np.maximum.accumulate((1 + un.net_ex.loc[a:b] + rf.loc[a:b]).cumprod()) - 1).min(),
                     "turnover": sim.turnover.loc[a:b].mean()})
        store[(s, name)] = (x.to_numpy(), R.to_numpy(), Q.to_numpy(), net.to_numpy())

T = pd.DataFrame(rows)
T.to_csv("dfl_diagnosis/panel_results.csv", index=False)
pd.set_option("display.width", 250)

print("== PANEL MEDIANS / MEANS across 16 series, dev 2000-2024 (exposure normalised to mean 1 for VR, VT, MT)")
g = T.groupby("forecaster")
print(pd.concat([g[["QLIKE", "VR_rv", "VT", "MT_alpha_pa", "SR_net", "timing_gain", "MaxDD", "turnover"]].mean(),
                 g["MT"].median().rename("MT_median"),
                 g["SR_net"].apply(lambda z: (z > T[T.forecaster == "lagged_rv"].set_index("series").SR_un.reindex(
                     T.loc[z.index, "series"]).to_numpy()).sum()).rename("n_beat_unmanaged_SR")], axis=1).round(4).to_string())

# paired comparisons with a panel block bootstrap (same month blocks across series)
rng = np.random.default_rng(0)
n = len(next(iter(store.values()))[0])
starts = rng.integers(0, n, (B, int(np.ceil(n / BLOCK))))
idx = ((starts[:, :, None] + np.arange(BLOCK)) % n).reshape(B, -1)[:, :n]


def stats(x, R, Q, net, I):
    xb, Rb, Qb, nb = x[I], R[I], Q[I], net[I]
    xb = xb / xb.mean(axis=1, keepdims=True)
    logvr = np.log((Qb * xb ** 2).mean(1) / Qb.mean(1))
    sr = nb.mean(1) / nb.std(1, ddof=1) * np.sqrt(12)
    mt = (xb * Rb).mean(1) / Rb.mean(1)
    return logvr, sr


def compare(A, Bn, model_filter=None):
    dv, ds, pv, ps = [], [], [], []
    sers = [s for s in SERIES if (s, A) in store and (s, Bn) in store]
    acc_v, acc_s = np.zeros(B), np.zeros(B)
    pt_v, pt_s = [], []
    for s in sers:
        va, sa = stats(*store[(s, A)], idx)
        vb, sb = stats(*store[(s, Bn)], idx)
        acc_v += va - vb; acc_s += sa - sb
        I0 = np.arange(n)[None, :]
        v0a, s0a = stats(*store[(s, A)], I0); v0b, s0b = stats(*store[(s, Bn)], I0)
        pt_v.append((v0a - v0b)[0]); pt_s.append((s0a - s0b)[0])
    k = len(sers)
    mv, msr = np.mean(pt_v), np.mean(pt_s)
    bv, bs = acc_v / k, acc_s / k
    p_v = np.mean(np.abs(bv - bv.mean()) >= abs(mv))
    p_s = np.mean(np.abs(bs - bs.mean()) >= abs(msr))
    return {"A": A, "B": Bn, "n_series": k,
            "mean dlogVR (A-B)": mv, "VR CI lo": np.percentile(bv - bv.mean() + mv, 2.5),
            "VR CI hi": np.percentile(bv - bv.mean() + mv, 97.5), "p_VR": p_v,
            "series with lower VR": int(np.sum(np.array(pt_v) < 0)),
            "mean dSR net (A-B)": msr, "SR CI lo": np.percentile(bs - bs.mean() + msr, 2.5),
            "SR CI hi": np.percentile(bs - bs.mean() + msr, 97.5), "p_SR": p_s,
            "series with higher SR": int(np.sum(np.array(pt_s) > 0))}


pairs = [("garch_risk", "garch_qmle"), ("garch_util", "garch_qmle"), ("garch_risk", "garch_util"),
         ("gjr_risk", "gjr_qmle"), ("gjr_util", "gjr_qmle"), ("gjr_risk", "gjr_util"),
         ("garch_qmle", "lagged_rv"), ("gjr_qmle", "garch_qmle")]
C = pd.DataFrame([compare(*p) for p in pairs])
C.to_csv("dfl_diagnosis/panel_tests.csv", index=False)
print("\n== PAIRED PANEL TESTS, dev 2000-2024, block bootstrap (6m blocks, common across series, 2000 reps)")
print(C.round(4).to_string(index=False))
