"""6. International check #1: JKP Global Factor Data (jkpfactors.com), 8 regions.

Closest JKP analogues of the US factors: Mkt-RF -> mkt, RMW -> ope_be (operating profits / book equity),
CMA -> at_gr1 (asset growth, low minus high), Mom -> ret_12_1. JKP factors are capped value-weighted tercile long-shorts per
country, in USD, so they are close but not identical to Fama-French factors. Same rule and settings as script 5.
GARCH first refit in January of max(1990, 2 full years after the mix starts). Regions: world ex US, Japan, UK, Germany, France,
Canada, Australia, emerging. Pool = the 7 non-overlapping regions (world ex US contains the others).
Tests: timing gain per region and pooled (mean over the pool, blocks common to all regions); Cederburg real-time combination
over the window where every region has 120 months of managed history.
"""
import numpy as np
import pandas as pd

import config
from core import data, report
from core.garch import garch_monthly
from core.rule import mm_weights, run
from core.stats import sharpe, perf, boot_idx, sr_rows, dsr_test, fmt, cederburg

S = report.start("6_intl_jkp", "6. International check #1: JKP factors, 8 regions")
REGIONS = ["world_ex_us", "jpn", "gbr", "deu", "fra", "can", "aus", "emerging"]
POOL = [r for r in REGIONS if r != "world_ex_us"]
MIXES = {"Mkt + Prof + Inv + Mom": ["mkt", "ope_be", "at_gr1", "ret_12_1"],
         "Prof + Inv + Mom": ["ope_be", "at_gr1", "ret_12_1"],
         "Mkt + Mom": ["mkt", "ret_12_1"],
         "Market alone": ["mkt"],
         "Mkt + Prof + Inv (no Mom)": ["mkt", "ope_be", "at_gr1"],
         "Mom alone": ["ret_12_1"]}
rf = data.rf()
nets, combos, info = {}, {}, []
for reg in REGIONS:
    D, M = data.jkp(reg)
    for mix, fs in MIXES.items():
        e = M[fs].dropna().mean(axis=1).loc[:"2025-12"]; start = e.index[0]
        d = D[fs].loc[D[fs].dropna().index[0]:].fillna(0.0).mean(axis=1)          # a missing day of one factor = 0 that day
        fy = max(config.INTL_FIRST_GARCH_YEAR, int(start[:4]) + 2 + (start[5:] != "01"))
        v = garch_monthly(f"jkp_{reg}_" + "_".join(fs), d, first_year=fy).reindex(e.index)
        w, _ = mm_weights(v, e, S, (f"{fy}-01", "1997-12"))
        z = run(w, e, rf, S["cost"])[0]
        nets[(reg, mix, "MM on GARCH")], nets[(reg, mix, "Unmanaged")] = z, e.loc[z.index]
        combos[(reg, mix)] = cederburg(e.loc[z.index], z)[0]
        if mix == "Mkt + Prof + Inv + Mom":
            info.append({"region": reg, "data from": start, "first GARCH refit": fy, "managed from": z.index[0],
                         "mean w 1998+": round(w.loc["1998-01":].mean(), 2)})
print("Data and calibration windows (Mkt + Prof + Inv + Mom)"); report.show(pd.DataFrame(info).set_index("region"))
c_start = max(c.index[0] for c in combos.values())
print(f"Cederburg combination window: {c_start} to 2025-12 (first month with 120 months of managed history in every region)\n")


def pooled(mix, a, b, key):
    get = (lambda r: nets[(r, mix, "MM on GARCH")]) if key == "MM" else (lambda r: combos[(r, mix)])
    A = np.stack([get(r).loc[a:b].to_numpy() for r in POOL]); U = np.stack([nets[(r, mix, "Unmanaged")].loc[a:b].to_numpy() for r in POOL])
    I = boot_idx(A.shape[1]); g0 = np.array([sharpe(x) - sharpe(y) for x, y in zip(A, U)])
    gb = np.mean([sr_rows(x[I]) - sr_rows(y[I]) for x, y in zip(A, U)], axis=0)
    lo, hi = np.percentile(gb - gb.mean() + g0.mean(), [2.5, 97.5])
    return f"{fmt(g0.mean(), lo, hi)}  {int((g0 > 0).sum())}/{len(POOL)}"


rows, summary = [], {}
for mix in MIXES:
    print(f"================ {mix}")
    for p, (a, b) in config.PERIODS.items():
        T = {}
        for reg in REGIONS:
            z, u = nets[(reg, mix, "MM on GARCH")].loc[a:b], nets[(reg, mix, "Unmanaged")].loc[a:b]
            pm, pu = perf(z, rf.loc[a:b]), perf(u, rf.loc[a:b]); d, lo, hi, pv, _ = dsr_test(z, u)
            T[reg] = {"Sharpe MM": pm["Sharpe"], "Sharpe unm": pu["Sharpe"], "Sortino MM": pm["Sortino"], "Sortino unm": pu["Sortino"],
                      "MaxDD MM": pm["MaxDD_%"], "MaxDD unm": pu["MaxDD_%"], "timing gain [95% CI]": fmt(d, lo, hi)}
            rows.append({"mix": mix, "period": p, "region": reg, **T[reg], "p": pv})
        print(f"--- {p}"); report.show(pd.DataFrame(T).T)
        summary.setdefault(mix, {})[p] = pooled(mix, a, b, "MM")
        print(f"pooled 7 regions, timing gain: {summary[mix][p]}")
        cs = max(a, c_start)
        ck = f"Cederburg {cs[:4]}-{b[:4]}"; summary[mix][ck] = pooled(mix, cs, b, "C")
        print(f"pooled 7 regions, Cederburg gain ({cs} to {b}): {summary[mix][ck]}\n")
pd.DataFrame(rows).to_csv(config.out_dir() / "6_intl_jkp.csv", index=False)
print("SUMMARY: pooled mean Sharpe gain [95% CI], regions that gain (out of 7)")
X = pd.DataFrame(summary).T
report.show(X[list(config.PERIODS)]); report.show(X[[c for c in X.columns if c.startswith("Cederburg")]])
report.save_nets("6_intl_jkp", {(f"JKP {reg}", mix, rule): z for (reg, mix, rule), z in nets.items()})
