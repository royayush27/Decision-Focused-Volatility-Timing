"""4. US, which factor combination is best, and industry groups.

Part A: all 63 equal-weight subsets of the six factors (no timing). Picking the best one after seeing 1998-2025 is hindsight,
        so the subset table is context, not a strategy. Real-time picks for comparison: the best 3 on 1963-1997 (then fixed),
        and the top 3 re-chosen every January (expanding window, or trailing 5 years).
Part A2: Sharpe and rank of selected mixes among all 63, before and after 1998 (gross).
Part B: industry groups (equal weights): defensive (NoDur, Hlth, Utils, Telcm), cyclical (Durbl, Manuf, Enrgy, HiTec, Shops,
        Other) and all 10, under the timing rules of script 2.
"""
from itertools import combinations

import numpy as np
import pandas as pd

import config
from core import data, report
from core.garch import garch_monthly
from core.rule import mm_weights, run, run_multi
from core.stats import sharpe, boot_idx, dsr_test, fmt
from core.variants import xu, conditional, recovery, rv_states

S = report.start("4_us_factor_combos", "4. US, factor combinations and industry groups")
FW = ("1970-01", "1997-12")
F = data.FACTORS
R = data.us_monthly(F); rf = data.rf().reindex(R.index)
idx = R.loc["1998-01":].index


def fixed(names, ix=idx):
    return pd.DataFrame({f: (1 / len(names) if f in names else 0.0) for f in F}, index=ix)


def top3(window):
    W = pd.DataFrame(0.0, index=idx, columns=F)
    for t in idx:
        hist = R.loc[:t[:4] + "-01"].iloc[:-1]
        hist = hist if window is None else hist.iloc[-window:]
        W.loc[t, (hist.mean() / hist.std()).nlargest(3).index] = 1 / 3
    return W


net = lambda W: run_multi(W, R, S["cost"])[0]
pre = R.loc[:"1997-12"]; best_pre = list((pre.mean() / pre.std()).nlargest(3).index)
named = {"all six (EW)": fixed(F), "Mkt + RMW + CMA + Mom (hindsight pick)": fixed(["Mkt-RF", "RMW", "CMA", "Mom"]),
         "RMW + CMA + Mom (hindsight pick)": fixed(["RMW", "CMA", "Mom"]),
         f"best 3 on 1963-1997 ({', '.join(best_pre)})": fixed(best_pre),
         "top 3, re-chosen yearly (expanding)": top3(None), "top 3, re-chosen yearly (trailing 5y)": top3(60),
         "market alone": fixed(["Mkt-RF"])}
NN = {k: net(W) for k, W in named.items()}
subsets = {c: net(fixed(list(c))) for k in range(1, 7) for c in combinations(F, k)}
print("Part A. Equal-weight factor mixes, net Sharpe, and percentile among all 63 subsets")
T = pd.DataFrame({k: {p: sharpe(z.loc[a:b]) for p, (a, b) in config.PERIODS.items()} for k, z in NN.items()}).T
for p, (a, b) in config.PERIODS.items():
    dist = np.array([sharpe(z.loc[a:b]) for z in subsets.values()])
    T[f"{p} pctile"] = [100 * (dist < v).mean() for v in T[p]]
report.show(T)
for p, (a, b) in config.PERIODS.items():
    s = pd.Series({"+".join(c): sharpe(z.loc[a:b]) for c, z in subsets.items()})
    print(f"  {p}: median subset {s.median():.2f}; best {s.idxmax()} {s.max():.2f}; worst {s.idxmin()} {s.min():.2f}")
a1 = pd.Series({"+".join(c): sharpe(z.loc["1998-01":"2015-12"]) for c, z in subsets.items()})
a2 = pd.Series({"+".join(c): sharpe(z.loc["2016-01":"2025-12"]) for c, z in subsets.items()})
print(f"  rank correlation of subset Sharpe ratios, 1998-2015 vs 2016-2025: {a1.rank().corr(a2.rank()):.2f}\n")

print("Part A2. Gross Sharpe and rank among all 63 equal-weight mixes, before and after 1998")
print("  (RMW and CMA were published in 2013-2015 with data up to then: before 2016 is not out of sample for them)")
P2 = {"1963-1997": ("1963-07", "1997-12"), "1963-1980": ("1963-07", "1980-12"), "1981-1997": ("1981-01", "1997-12"), **config.PERIODS}
raw = {"+".join(c): R[list(c)].mean(axis=1) for k in range(1, 7) for c in combinations(F, k)}
rows = {}
for p, (a, b) in P2.items():
    s = pd.Series({k: sharpe(z.loc[a:b]) for k, z in raw.items()}); rk = s.rank(ascending=False)
    for k in ("Mkt-RF+RMW+CMA+Mom", "RMW+CMA+Mom", "Mkt-RF+SMB+HML+RMW+CMA+Mom", "Mkt-RF"):
        rows.setdefault(k, {})[p] = f"{s[k]:.2f} (#{int(rk[k])})"
    rows.setdefault("best mix", {})[p] = f"{s.idxmax()} {s.max():.2f}"
report.show(pd.DataFrame(rows).T)

print("Part B. Industry groups (equal weights), net Sharpe (gain vs unmanaged [95% CI])")
GROUPS = {"defensive": ["NoDur", "Hlth", "Utils", "Telcm"], "cyclical": ["Durbl", "Manuf", "Enrgy", "HiTec", "Shops", "Other"],
          "all10": data.INDUSTRIES}
EI = data.us_monthly(data.INDUSTRIES); DI = data.us_daily(data.INDUSTRIES)
nn = {}
for g, inds in GROUPS.items():
    e = EI[inds].mean(axis=1); rd = DI[inds].dropna().mean(axis=1)
    rv = (rd ** 2).groupby(rd.index.to_period("M")).sum(); rv.index = rv.index.astype(str); rv1 = rv.shift(1).reindex(e.index)
    rv5, rv63 = rv_states(rd, e.index)
    vg = garch_monthly(f"us_grp_{g}", rd, config.US_FIRST_GARCH_YEAR).reindex(e.index)
    W = {"Unmanaged": pd.Series(1.0, index=e.index)}
    W["MM on RV"], _ = mm_weights(rv1, e, S, FW); W["MM on GARCH"], _ = mm_weights(vg, e, S, FW)
    W["Xu switches (GARCH)"] = xu(W["MM on GARCH"], e, rv1, S)
    W["Conditional quintiles (GARCH)"] = conditional(W["MM on GARCH"], rv1, S)
    W["Recovery state (RV)"] = recovery(W["MM on RV"], rv1, rv5, rv63, S)
    nets = {k: run(w.loc["1970-01":], e, rf, S["cost"])[0] for k, w in W.items()}
    out = {}
    for p, (a, b) in config.PERIODS.items():
        u = nets["Unmanaged"].loc[a:b]; I = boot_idx(len(u)); col = {}
        for k, z in nets.items():
            if k == "Unmanaged":
                col[k] = f"{sharpe(u):.2f}"; continue
            d, lo, hi, pv, _ = dsr_test(z.loc[a:b], u, I); col[k] = f"{sharpe(z.loc[a:b]):.2f} ({fmt(d, lo, hi)})"
        out[p] = col
    print(f"\n  {g}: {', '.join(inds)}"); report.show(pd.DataFrame(out))
    nn.update({("US industry group", g, k): z for k, z in nets.items()})
report.save_nets("4_us_factor_combos", nn)
