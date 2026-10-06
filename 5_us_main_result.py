"""5. US, the main result: equal-weight factor mixes, timed with Moreira-Muir on a GARCH(1,1) of the mix itself, vs unmanaged.

Each mix = equal weights in its factors, rebalanced monthly. MM on GARCH: a real-time GARCH(1,1) on the mix's daily returns
gives the variance forecast; exposure w = c / v_hat with the spec's c rule, cap and costs.
Reported per period: Sharpe, Sortino, CAGR, vol, max drawdown, as is and levered to 15% annual vol (constant leverage set on
pre-1998 data only; borrowing at the risk-free rate, optimistic for long-short factors); Sharpe gain from timing with a
block-bootstrap CI; Cederburg real-time combination gain.
Note: Mkt + RMW + CMA + Mom and Mkt + RMW + Mom were named AFTER seeing 1998-2025 results (hindsight; see script 4 and 8).
Writes results/<spec>/5_us_main_result.json for the equity / drawdown dashboard.
"""
import json

import numpy as np
import pandas as pd

import config
from core import data, report
from core.garch import garch_monthly
from core.rule import mm_weights, run
from core.stats import perf, boot_idx, dsr_test, fmt, cederburg

S = report.start("5_us_main_result", "5. US main result: equal-weight factor mixes, timed vs unmanaged")
FW = ("1970-01", "1997-12")
MIXES = {"All six": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
         "Mkt + RMW + CMA + Mom": ["Mkt-RF", "RMW", "CMA", "Mom"],
         "Mkt + RMW + Mom": ["Mkt-RF", "RMW", "Mom"],
         "FF5 (no Mom)": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
         "FF3 (Mkt + SMB + HML)": ["Mkt-RF", "SMB", "HML"],
         "No market (SMB + HML + RMW + CMA + Mom)": ["SMB", "HML", "RMW", "CMA", "Mom"],
         "RMW + CMA + Mom": ["RMW", "CMA", "Mom"],
         "Mkt + Mom": ["Mkt-RF", "Mom"],
         "Market alone": ["Mkt-RF"]}
R = data.us_monthly(data.FACTORS); rf = data.rf().reindex(R.index); D = data.us_daily(data.FACTORS)


def cagr(z, r):
    return 100 * (np.prod(1 + z + r) ** (12 / len(z)) - 1)


series, lever, rows, nn = {}, {}, [], {}
for name, fs in MIXES.items():
    e = R[fs].mean(axis=1)
    v = garch_monthly("us_mix_" + "_".join(f.replace("-", "") for f in fs), D[fs].mean(axis=1).dropna(),
                      config.US_FIRST_GARCH_YEAR).reindex(e.index)
    w, _ = mm_weights(v, e, S, FW)
    z_mm = run(w.loc["1970-01":], e, rf, S["cost"])[0]; z_u = e.loc[z_mm.index]
    combo = cederburg(z_u, z_mm)[0]
    for ver, z in (("MM-GARCH", z_mm), ("Unmanaged", z_u)):
        series[(name, ver)] = z; lever[(name, ver)] = 0.15 / (z.loc[:"1997-12"].std() * np.sqrt(12))
    nn[("US mix", name, "Unmanaged")] = z_u; nn[("US mix", name, "MM on GARCH")] = z_mm
    for p, (a, b) in config.PERIODS.items():
        I = boot_idx(len(z_u.loc[a:b])); d, lo, hi, pv, _ = dsr_test(z_mm.loc[a:b], z_u.loc[a:b], I)
        dc, lc, hc, pc, _ = dsr_test(combo.loc[a:b], z_u.loc[a:b], I)
        for ver in ("MM-GARCH", "Unmanaged"):
            z = series[(name, ver)].loc[a:b]
            for view, k in (("as is", 1.0), ("levered to 15% vol", lever[(name, ver)])):
                rows.append({"mix": name, "version": ver, "period": p, "view": view, **perf(k * z, rf.loc[a:b]),
                             "CAGR_%": cagr(k * z, rf.loc[a:b]),
                             "timing gain [95% CI]": fmt(d, lo, hi) if ver == "MM-GARCH" else "", "p": pv if ver == "MM-GARCH" else np.nan,
                             "Cederburg gain [95% CI]": fmt(dc, lc, hc) if ver == "MM-GARCH" else ""})
T = pd.DataFrame(rows); T.to_csv(config.out_dir() / "5_us_main_result.csv", index=False)
for p in config.PERIODS:
    for view in ("as is", "levered to 15% vol"):
        X = T[(T.period == p) & (T.view == view)]
        P = X.pivot_table(index="mix", columns="version", values=["Sharpe", "Sortino", "CAGR_%", "MaxDD_%"]).loc[list(MIXES)]
        P.columns = [f"{a} {'MM' if b == 'MM-GARCH' else 'unm'}" for a, b in P.columns]
        if view == "as is":
            G = X[X.version == "MM-GARCH"].set_index("mix").loc[list(MIXES), ["timing gain [95% CI]", "p", "Cederburg gain [95% CI]"]]
            P = P.round(2).join(G)
        print(f"======== {p}, {view}"); report.show(P)
out = pd.DataFrame({f"{n} | {v}": z.loc["1998-01":"2025-12"] for (n, v), z in series.items()})
json.dump({"spec": S["name"], "months": list(out.index), "rf": list(rf.loc["1998-01":"2025-12"].round(8)),
           "series": {f"{n}|{v}": list(series[(n, v)].loc["1998-01":"2025-12"].round(8)) for (n, v) in series},
           "lever": {f"{n}|{v}": round(L, 4) for (n, v), L in lever.items()}, "mixes": list(MIXES)},
          open(config.out_dir() / "5_us_main_result.json", "w"))
report.save_nets("5_us_main_result", nn)
