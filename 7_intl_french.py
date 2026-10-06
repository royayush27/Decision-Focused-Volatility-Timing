"""7. International check #2: Ken French regional factors (same definitions as the US factors), plus leg-wise timing.

Regions: Developed ex US (headline), Europe, Japan, Asia Pacific ex Japan, North America, Developed, and the US for reference.
Data July 1990 - Aug 2026 (momentum from Nov 1990), so GARCH refits start in January 1993 (US: 1970).
Pool = Europe + Japan + Asia Pacific ex Japan (non-overlapping and non-US).
Three ways to time a mix (same rule and settings as script 5):
  Whole mix               one GARCH on the mix's daily returns, one exposure for the whole mix (as in scripts 5 and 6)
  Legs, mkt unmanaged     the market leg held at 1, every other leg vol-managed on its own GARCH, then equal weights
  Legs, all managed       every leg vol-managed on its own GARCH, then equal weights
The leg-wise designs were chosen after seeing script 6 (post-hoc). Cederburg combination for the whole-mix rule.
"""
import numpy as np
import pandas as pd

import config
from core import data, report
from core.garch import garch_monthly
from core.rule import mm_weights, run
from core.stats import sharpe, maxdd, boot_idx, sr_rows, dsr_test, fmt, cederburg

S = report.start("7_intl_french", "7. International check #2: Ken French regional factors, plus leg-wise timing")
REGIONS = ["Developed_ex_US", "Europe", "Japan", "Asia_Pacific_ex_Japan", "North_America", "Developed", "US"]
POOL = ["Europe", "Japan", "Asia_Pacific_ex_Japan"]
MIXES = {"Mkt + RMW + CMA + Mom": ["Mkt-RF", "RMW", "CMA", "Mom"],
         "RMW + CMA + Mom": ["RMW", "CMA", "Mom"],
         "Mkt + Mom": ["Mkt-RF", "Mom"],
         "Mkt + RMW + CMA (no Mom)": ["Mkt-RF", "RMW", "CMA"],
         "Market alone": ["Mkt-RF"],
         "Mom alone": ["Mom"],
         "All six": data.FACTORS}
VERS = ["Unmanaged", "Whole mix", "Legs, mkt unmanaged", "Legs, all managed"]
F = data.FACTORS; rf = data.rf()
tag = lambda fs: "_".join(f.replace("-", "") for f in fs)


def load(reg):
    if reg == "US":
        return data.us_daily(F), data.us_monthly(F), config.US_FIRST_GARCH_YEAR, "us_", "us_mix_"
    D, M = data.ff_intl(reg)
    return D, M, 1993, f"ffi_{reg}_", f"ffi_{reg}_"


nets, combos = {}, {}
for reg in REGIONS:
    D, M, fy, pre_leg, pre_mix = load(reg)
    M = M.dropna().loc[:"2025-12"]; D = D.loc[D.dropna().index[0]:]; fw = (f"{fy}-01", "1997-12")

    def managed(name, e, d):
        v = garch_monthly(name, d, fy).reindex(e.index)
        return run(mm_weights(v, e, S, fw)[0], e, rf, S["cost"])[0]

    leg = {f: managed(pre_leg + (f if reg == "US" else f.replace("-", "")), M[f], (D[f] if reg == "US" else D[f].fillna(0.0)))
           for f in F}
    for mix, fs in MIXES.items():
        e = M[fs].mean(axis=1)
        wm = managed(pre_mix + tag(fs), e, D[fs].mean(axis=1).dropna() if reg == "US" else D[fs].fillna(0.0).mean(axis=1))
        idx = wm.index.intersection(pd.concat([leg[f] for f in fs], axis=1).dropna().index)
        lm = pd.concat([M[f] if f == "Mkt-RF" else leg[f] for f in fs], axis=1).loc[idx].mean(axis=1)
        la = pd.concat([leg[f] for f in fs], axis=1).loc[idx].mean(axis=1)
        for ver, z in zip(VERS, (e.loc[idx], wm.loc[idx], lm, la)):
            nets[(reg, mix, ver)] = z
        combos[(reg, mix)] = cederburg(e.loc[idx], wm.loc[idx])[0]
c_start = max(c.index[0] for (r, m), c in combos.items() if r != "US")
print(f"Cederburg combination window (international): {c_start} to 2025-12\n")


def gain(r, mix, ver, a, b):
    A = (combos[(r, mix)] if ver == "Cederburg" else nets[(r, mix, ver)]).loc[a:b].to_numpy()
    U = nets[(r, mix, "Unmanaged")].loc[a:b].to_numpy()
    return A, U


rows, summary = [], {}
for mix in MIXES:
    print(f"================ {mix}")
    for p, (a, b) in config.PERIODS.items():
        T = pd.DataFrame({ver: {r: sharpe(nets[(r, mix, ver)].loc[a:b]) for r in REGIONS} for ver in VERS})
        T.columns = ["SR unm", "SR whole", "SR legs mkt unm", "SR legs all"]
        for ver, col in (("Unmanaged", "DD unm"), ("Whole mix", "DD whole")):
            T[col] = [100 * maxdd(nets[(r, mix, ver)].loc[a:b], rf.loc[a:b]) for r in REGIONS]
        print(f"--- {p} (net Sharpe; max drawdown %)"); report.show(T)
        for ver in VERS[1:] + ["Cederburg"]:
            aa = max(a, c_start) if ver == "Cederburg" else a
            AU = [gain(r, mix, ver, aa, b) for r in POOL]; I = boot_idx(len(AU[0][0]))
            g0 = np.array([sharpe(x) - sharpe(y) for x, y in AU]); gb = np.mean([sr_rows(x[I]) - sr_rows(y[I]) for x, y in AU], axis=0)
            lo, hi = np.percentile(gb - gb.mean() + g0.mean(), [2.5, 97.5])
            one = []
            for r in ("Developed_ex_US", "US"):
                ar = aa if (ver != "Cederburg" or r != "US") else a
                x, y = gain(r, mix, ver, ar, b)
                if len(x) == len(y) and len(x) > 24:
                    d, l2, h2, pv, _ = dsr_test(x, y); one.append(f"{r[:12]} {fmt(d, l2, h2)}")
            label = f"{ver} ({aa[:4]}-{b[:4]})" if ver == "Cederburg" else ver
            s = f"{fmt(g0.mean(), lo, hi)} ({int((g0 > 0).sum())}/3)"
            summary.setdefault((mix, ver), {})[p] = s
            print(f"  gain {label:28s} pooled 3: {s} | " + "  ".join(one))
            rows.append({"mix": mix, "period": p, "version": ver, "pooled gain": s, "single": " | ".join(one)})
        print()
pd.DataFrame(rows).to_csv(config.out_dir() / "7_intl_french.csv", index=False)
print("SUMMARY: pooled 3 regions, mean Sharpe gain vs unmanaged [95% CI] (regions that gain)")
X = pd.DataFrame(summary).T; X.index = [f"{m} | {v}" for m, v in X.index]; report.show(X)
report.save_nets("7_intl_french", {(f"French {reg}", mix, ver): z for (reg, mix, ver), z in nets.items() if reg != "US"})
