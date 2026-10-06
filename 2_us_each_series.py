"""2. US, each series on its own: market, SMB, HML, RMW, CMA, Mom and the 10 industry portfolios (16 series).

Same rules as script 1 minus the VIX one (VIX is a market-level quantity). Each series has its own real-time GARCH(1,1).
Reported per period: mean net Sharpe across series (all 16 / 6 factors / 10 industries); mean Sharpe gain vs unmanaged with a
block bootstrap whose blocks are common to all series, and the number of series that gain; the same for the Cederburg
combination of MM on GARCH; per-series Sharpe ratios and max drawdowns.
"""
import numpy as np
import pandas as pd

import config
from core import data, report
from core.garch import garch_monthly
from core.rule import mm_weights, run
from core.stats import sharpe, maxdd, mt_vt, boot_idx, sr_rows, fmt, cederburg
from core.variants import xu, conditional, recovery, rv_states

S = report.start("2_us_each_series", "2. US, each of the 16 series on its own")
FW = ("1970-01", "1997-12")
SER, FAC, IND = data.SERIES, data.FACTORS, data.INDUSTRIES
RULES = ["Unmanaged", "MM on RV", "MM on GARCH", "Xu switches (GARCH)", "Conditional quintiles (GARCH)", "Recovery state (RV)"]

E = data.us_monthly(SER); RV = data.us_rv(SER); D = data.us_daily(SER); rf = data.rf().reindex(E.index)
nets, combo, Wall = {}, {}, {}
for s in SER:
    e, rv1, r = E[s], RV[s].shift(1), D[s].dropna()
    vg = garch_monthly(f"us_{s}", r, config.US_FIRST_GARCH_YEAR).reindex(E.index); rv5, rv63 = rv_states(r, E.index)
    W = {"Unmanaged": pd.Series(1.0, index=E.index)}
    W["MM on RV"], _ = mm_weights(rv1, e, S, FW); W["MM on GARCH"], _ = mm_weights(vg, e, S, FW)
    W["Xu switches (GARCH)"] = xu(W["MM on GARCH"], e, rv1, S)
    W["Conditional quintiles (GARCH)"] = conditional(W["MM on GARCH"], rv1, S)
    W["Recovery state (RV)"] = recovery(W["MM on RV"], rv1, rv5, rv63, S)
    for k, w in W.items():
        nets[(k, s)] = run(w.loc["1970-01":], e, rf, S["cost"])[0]; Wall[(k, s)] = w
    combo[s] = cederburg(nets[("Unmanaged", s)], nets[("MM on GARCH", s)])[0]

rows = []
for p, (a, b) in config.PERIODS.items():
    I = boot_idx(len(nets[("Unmanaged", "Mkt-RF")].loc[a:b]))
    for k in RULES:
        for s in SER:
            z = nets[(k, s)].loc[a:b]; w = Wall[(k, s)].loc[a:b]; mt, vt = mt_vt(w, E[s].loc[a:b])
            rows.append({"period": p, "rule": k, "series": s, "Sharpe": sharpe(z), "MaxDD_%": 100 * maxdd(z, rf.loc[a:b]), "MT": mt, "VT": vt})
    T = pd.DataFrame(rows); T = T[T.period == p]; P = T.pivot(index="series", columns="rule", values="Sharpe")
    print(f"======== {p}")
    print("Mean net Sharpe across series")
    report.show(pd.DataFrame({"all 16": P.mean(), "6 factors": P.loc[FAC].mean(), "10 industries": P.loc[IND].mean()}).loc[RULES])
    out = {}
    for k in RULES[1:] + ["Cederburg (MM on GARCH)"]:
        for g, ss in (("all 16", SER), ("6 factors", FAC), ("10 industries", IND)):
            if k.startswith("Cederburg"):
                if any(len(combo[s].loc[a:b]) != len(I[0]) for s in ss):
                    continue
                A = {s: combo[s].loc[a:b].to_numpy() for s in ss}
            else:
                A = {s: nets[(k, s)].loc[a:b].to_numpy() for s in ss}
            U = {s: nets[("Unmanaged", s)].loc[a:b].to_numpy() for s in ss}
            d_s = np.array([sharpe(A[s]) - sharpe(U[s]) for s in ss])
            db = np.mean([sr_rows(A[s][I]) - sr_rows(U[s][I]) for s in ss], axis=0)
            lo, hi = np.percentile(db - db.mean() + d_s.mean(), [2.5, 97.5])
            out.setdefault(k, {})[g] = f"{fmt(d_s.mean(), lo, hi)}  {int((d_s > 0).sum())}/{len(ss)}"
    print("Mean Sharpe gain vs unmanaged [95% CI], series that gain")
    report.show(pd.DataFrame(out).T)
    print("Net Sharpe by series"); report.show(P[RULES])
pd.DataFrame(rows).to_csv(config.out_dir() / "2_us_each_series.csv", index=False)
T = pd.DataFrame(rows)
X = T[T.period == config.MAIN].pivot(index="series", columns="rule", values="MaxDD_%")
print(f"Max drawdown %, {config.MAIN}"); report.show(X[["Unmanaged", "MM on RV", "MM on GARCH"]], 1)
report.save_nets("2_us_each_series", {("US series", s, k): nets[(k, s)] for k in RULES for s in SER})
