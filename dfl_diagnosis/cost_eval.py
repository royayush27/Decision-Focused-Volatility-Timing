import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd
from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
B, BL = 2000, 6
rng = np.random.default_rng(1)

for kbp in (25, 50):
    kappa = kbp / 1e4
    rows, nets = [], {}
    for s in SERIES:
        ms = m[m["series"] == s].set_index("month").sort_index().loc["1970-01":"2025-12"]
        rf = rf_all.reindex(ms.index)
        ex = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
        rv = ms["rv"]
        d = pd.read_csv(f"dfl_diagnosis/panel/{s}_cost{kbp}.csv", index_col="month")
        fc = {"unmanaged": None, "qmle": d["qmle"], "cost_risk": d["cost_risk"]}
        for k, v in fc.items():
            w = pd.Series(1.0, index=ex.index) if v is None else pd.Series(exposure(v, solve_c(v.loc[:"1999-12"])), index=ex.index)
            sim = simulate(w, ex, rf, kappa); sim.index = ex.index
            z = sim.net_ex.loc["2000-01":"2024-12"]
            W = w.loc["2000-01":"2024-12"]; x = W / W.mean(); Q = rv.loc["2000-01":"2024-12"]
            nets[(k, s)] = z.to_numpy()
            rows.append({"series": s, "rule": k, "SR_net": z.mean() / z.std() * np.sqrt(12),
                         "turnover": sim.turnover.loc["2000-01":"2024-12"].mean(),
                         "cost_drag_pa": sim.cost.loc["2000-01":"2024-12"].mean() * 12,
                         "VR_rv": (Q * x ** 2).mean() / Q.mean()})
    T = pd.DataFrame(rows)
    print(f"\n===== cost {kbp} bps per unit turnover, dev 2000-2024, 16 series")
    print(T.groupby("rule")[["SR_net", "turnover", "cost_drag_pa", "VR_rv"]].mean().round(4).to_string())
    n = len(next(iter(nets.values())))
    st = rng.integers(0, n, (B, int(np.ceil(n / BL))))
    I = ((st[:, :, None] + np.arange(BL)) % n).reshape(B, -1)[:, :n]
    for A, Bn in (("cost_risk", "qmle"), ("cost_risk", "unmanaged"), ("qmle", "unmanaged")):
        dB = np.zeros(B); d0 = []
        for s in SERIES:
            a, b = nets[(A, s)], nets[(Bn, s)]
            dB += (a[I].mean(1) / a[I].std(1, ddof=1) - b[I].mean(1) / b[I].std(1, ddof=1)) * np.sqrt(12)
            d0.append((a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * np.sqrt(12))
        dB /= len(SERIES); m0 = np.mean(d0)
        print(f"  {A:9s} - {Bn:9s}: mean dSR {m0:+.4f}  CI [{np.percentile(dB-dB.mean()+m0,2.5):+.4f}, "
              f"{np.percentile(dB-dB.mean()+m0,97.5):+.4f}]  p={np.mean(np.abs(dB-dB.mean())>=abs(m0)):.3f}  "
              f"series better {int(np.sum(np.array(d0)>0))}/16")
    T.to_csv(f"dfl_diagnosis/cost{kbp}_results.csv", index=False)
