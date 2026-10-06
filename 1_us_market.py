"""1. US market (Mkt-RF): which volatility-timing rule works?

Rules (all real time; exposure w = c / v with the c rule and cap of the spec):
  Unmanaged          w = 1
  MM on RV           v = last month's realised variance (Moreira-Muir 2017)
  MM on GARCH        v = GARCH(1,1) forecast of this month's variance          <- the main rule of the project
  MM on GJR          v = GJR-GARCH(1,1) forecast (robustness: better forecast, does it pay?)
  MM on VIX^2        v = VIX^2 at the end of last month (from 1990; best risk forecast, worst mean timing)
  Xu switches / Conditional quintiles / Recovery state: see core/variants.py
Reported per period: net Sharpe, Sortino, MT x VT decomposition, return, vol, max drawdown, average exposure, share of months
at the cap; Sharpe gain vs unmanaged with a block-bootstrap CI; Cederburg real-time combination gain vs unmanaged.
"""
import numpy as np
import pandas as pd

import config
from core import data, report
from core.garch import garch_monthly
from core.rule import mm_weights, run
from core.stats import perf, boot_idx, dsr_test, fmt, cederburg
from core.variants import xu, conditional, recovery, rv_states

S = report.start("1_us_market", "1. US market: which volatility-timing rule works?")
FW, FW_VIX = ("1970-01", "1997-12"), ("1993-01", "1997-12")      # frozen-c windows (spec 'old' only)

e = data.us_monthly(["Mkt-RF"])["Mkt-RF"]; rf = data.rf().reindex(e.index)
rv1 = data.us_rv(["Mkt-RF"])["Mkt-RF"].shift(1)
r = data.us_daily(["Mkt-RF"])["Mkt-RF"]
vg = garch_monthly("us_Mkt-RF", r, config.US_FIRST_GARCH_YEAR).reindex(e.index)
vj = garch_monthly("us_Mkt-RF", r, config.US_FIRST_GARCH_YEAR, gjr=True).reindex(e.index)
iv = data.vix_sq_monthly().shift(1).reindex(e.index)
rv5, rv63 = rv_states(r, e.index)

W = {"Unmanaged": pd.Series(1.0, index=e.index)}
W["MM on RV"], _ = mm_weights(rv1, e, S, FW)
W["MM on GARCH"], _ = mm_weights(vg, e, S, FW)
W["MM on GJR"], _ = mm_weights(vj, e, S, FW)
W["MM on VIX^2"], _ = mm_weights(iv, e, S, FW_VIX)
W["Xu switches (GARCH)"] = xu(W["MM on GARCH"], e, rv1, S)
W["Conditional quintiles (GARCH)"] = conditional(W["MM on GARCH"], rv1, S)
W["Recovery state (RV)"] = recovery(W["MM on RV"], rv1, rv5, rv63, S)

nets = {k: run(w.loc["1970-01":], e, rf, S["cost"])[0] for k, w in W.items()}
combo = {k: cederburg(nets["Unmanaged"], z)[0] for k, z in nets.items() if k != "Unmanaged"}
cap = S["cap"] or np.inf
rows = []
for p, (a, b) in config.PERIODS.items():
    u = nets["Unmanaged"].loc[a:b]; I = boot_idx(len(u)); T = {}
    for k, z in nets.items():
        z = z.loc[a:b]; w = W[k].loc[a:b]
        T[k] = {**perf(z, rf.loc[a:b], w, e.loc[a:b]), "at_cap_%": 100 * (w >= cap - 1e-9).mean()}
        if k != "Unmanaged":
            d, lo, hi, pv, _ = dsr_test(z, u, I); T[k]["dSR vs unmanaged [95% CI]"] = fmt(d, lo, hi); T[k]["p"] = pv
            c = combo[k].loc[a:b]
            if len(c) == len(u):
                d, lo, hi, pv, _ = dsr_test(c, u, I); T[k]["Cederburg gain [95% CI]"] = fmt(d, lo, hi); T[k]["p_ced"] = pv
        rows.append({"period": p, "rule": k, **T[k]})
    print(f"======== {p}: {len(u)} months")
    report.show(pd.DataFrame(T).T)
pd.DataFrame(rows).to_csv(config.out_dir() / "1_us_market.csv", index=False)
report.save_nets("1_us_market", {("US market", "Mkt-RF", k): z for k, z in nets.items()})
print("Cederburg: each month the investor mixes unmanaged and managed with mean-variance weights estimated on past months only\n"
      f"(at least {config.CEDERBURG_MIN}); the gain is the combination's Sharpe minus the unmanaged Sharpe.")
