"""Part A: smaller factor mixes. Part B: industry groups. Evaluation 1998-2025 and sub-periods, net of 5 bps.
Part A (six factors Mkt-RF, SMB, HML, RMW, CMA, Mom; equal weights within a mix, rebalanced monthly):
  - all 63 possible subsets, to show where any one mix sits in the distribution (picking the best one AFTER seeing
    1998-2025 is hindsight, so the subset table is context, not a strategy);
  - 'RMW + CMA + Mom': the mix suggested after seeing the factor_mix results (hindsight, labelled as such);
  - 'best 3 on 1963-1997': the three factors with the highest Sharpe ratio before 1998, then fixed (real time);
  - 'top 3, re-chosen yearly': each January, the three factors with the highest expanding-window Sharpe ratio (real time);
  - 'top 3, trailing 5y': same with the past 60 months only.
Part B (10 industries in two groups; standard defensive / cyclical split):
  defensive = NoDur (consumer non-durables), Hlth, Utils, Telcm;   cyclical = Durbl, Manuf, Enrgy, HiTec, Shops, Other.
  Equal-weight group portfolios (and all 10), monthly excess returns, daily returns for RV and GARCH (garch_tools), then the
  main_eval rules: unmanaged, MM on RV / GARCH, Xu switches, conditional quintiles, recovery state. c frozen on 1970-1997."""
import os, sys
from itertools import combinations
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from garch_tools import daily, garch_monthly

F = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
PERIODS = {"1998-2025": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"),
           "2016-2025": ("2016-01", "2025-12"), "2021-2025": ("2021-01", "2025-12")}
CAL = ("1970-01", "1997-12")
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
sr = lambda z: z.mean() / z.std() * np.sqrt(12) if z.std() > 0 else 0.0
pd.set_option("display.width", 220)

# ---------------------------------------------------------------- Part A
R = pd.DataFrame({f: m[m["series"] == f].set_index("month").sort_index()["ret_m"] for f in F}).loc["1963-07":"2025-12"]
idx = R.loc["1998-01":].index


def ew_net(W):
    W = W.reindex(idx); g = (W * R.loc[idx]).sum(axis=1); to = W.diff().abs().sum(axis=1).fillna(0)
    return g - 5e-4 * to


def fixed(names):
    return pd.DataFrame({f: (1 / len(names) if f in names else 0.0) for f in F}, index=idx)


def rolling_top3(window):
    W = pd.DataFrame(0.0, index=idx, columns=F)
    for t in idx:
        year_start = t[:4] + "-01"
        hist = R.loc[:year_start].iloc[:-1] if window is None else R.loc[:year_start].iloc[:-1].iloc[-window:]
        top = (hist.mean() / hist.std()).nlargest(3).index
        W.loc[t, top] = 1 / 3
    return W


pre = R.loc[:"1997-12"]; best_pre = list((pre.mean() / pre.std()).nlargest(3).index)
named = {"all six (EW)": fixed(F), "RMW + CMA + Mom (hindsight pick)": fixed(["RMW", "CMA", "Mom"]),
         f"best 3 on 1963-1997 ({', '.join(best_pre)})": fixed(best_pre),
         "top 3, re-chosen yearly (expanding)": rolling_top3(None), "top 3, re-chosen yearly (trailing 5y)": rolling_top3(60),
         "market alone": fixed(["Mkt-RF"])}
NN = {k: ew_net(W) for k, W in named.items()}
subsets = {c: ew_net(fixed(list(c))) for k in range(1, 7) for c in combinations(F, k)}
print("Part A. Factor mixes, equal weights within each mix, net Sharpe")
rows = {}
for k, z in NN.items():
    rows[k] = {p: sr(z.loc[a:b]) for p, (a, b) in PERIODS.items()}
T = pd.DataFrame(rows).T
for p, (a, b) in PERIODS.items():
    dist = np.array([sr(z.loc[a:b]) for z in subsets.values()])
    T[f"{p} pctile in 63 subsets"] = [100 * (dist < v).mean() for v in T[p]]
print(T.round(2).to_string())
for p, (a, b) in PERIODS.items():
    s = pd.Series({"+".join(c): sr(z.loc[a:b]) for c, z in subsets.items()})
    print(f"  {p}: median subset {s.median():.2f}; best {s.idxmax()} {s.max():.2f}; worst {s.idxmin()} {s.min():.2f}")
s_main = pd.Series({"+".join(c): sr(z.loc["1998-01":"2015-12"]) for c, z in subsets.items()})
s_late = pd.Series({"+".join(c): sr(z.loc["2016-01":"2025-12"]) for c, z in subsets.items()})
print(f"  rank correlation of subset Sharpe ratios, 1998-2015 vs 2016-2025: {s_main.rank().corr(s_late.rank()):.2f} "
      "(how well the past ranking of mixes predicts the next decade)")
W3 = rolling_top3(None)
print("  factors picked by 'top 3, re-chosen yearly (expanding)', share of years:",
      ", ".join(f"{f} {100 * (W3[f] > 0).groupby(W3.index.str[:4]).first().mean():.0f}%" for f in F))

# ---------------------------------------------------------------- Part B
GROUPS = {"defensive (NoDur, Hlth, Utils, Telcm)": ["NoDur", "Hlth", "Utils", "Telcm"],
          "cyclical (Durbl, Manuf, Enrgy, HiTec, Shops, Other)": ["Durbl", "Manuf", "Enrgy", "HiTec", "Shops", "Other"],
          "all 10 industries": ["NoDur", "Hlth", "Utils", "Telcm", "Durbl", "Manuf", "Enrgy", "HiTec", "Shops", "Other"]}
dd = daily()


def rolling_intercept(r, v, window=60):
    a = pd.Series(np.nan, index=r.index)
    for i in range(window, len(r)):
        y, x = r.iloc[i - window:i].to_numpy(), v.iloc[i - window:i].to_numpy(); ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 48:
            a.iloc[i] = np.polyfit(x[ok], y[ok], 1)[1]
    return a


def mm(v):
    return pd.Series(exposure(v, solve_c(v.loc[CAL[0]:CAL[1]])), index=v.index)


print("\nPart B. Industry groups (equal-weight portfolios), net Sharpe; in brackets dSR vs unmanaged [95% CI]")
rng = np.random.default_rng(0)
srb = lambda M: np.where(M.std(1, ddof=1) > 0, M.mean(1) / np.where(M.std(1, ddof=1) > 0, M.std(1, ddof=1), 1) * np.sqrt(12), 0.0)
for gname, inds in GROUPS.items():
    ms = pd.DataFrame({s: m[m["series"] == s].set_index("month").sort_index()["ret_m"] for s in inds}).loc["1963-07":"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms.mean(axis=1) - rf
    rd = pd.DataFrame({s: dd[dd["series"] == s].set_index("date")["ret"] for s in inds}).sort_index().mean(axis=1)
    rv = (rd ** 2).groupby(rd.index.to_period("M")).sum(); rv.index = rv.index.astype(str); rv1 = rv.shift(1).reindex(ms.index)
    eom = rd.groupby(rd.index.to_period("M")).apply(lambda z: z.index[-1])
    f = lambda k: pd.Series(((rd ** 2).rolling(k).sum() * 21 / k).loc[eom.values].to_numpy(), index=eom.index.astype(str)).shift(1).reindex(ms.index)
    rv5, rv63 = f(5), f(63)
    tag = "grp_" + gname.split(" ")[0] + ("" if "all" not in gname else "10")
    vg = garch_monthly(tag, rd).reindex(ms.index)
    W = {"Unmanaged": pd.Series(1.0, index=ms.index), "MM on RV": mm(rv1), "MM on GARCH": mm(vg)}
    hold = e.rolling(12).mean().shift(1) > 0; scale = rolling_intercept(e, rv1) > 0
    W["Xu switches (GARCH)"] = W["MM on GARCH"].where(scale, 1.0).where(hold, 0.0)
    q20, q80 = rv1.expanding(120).quantile(0.2), rv1.expanding(120).quantile(0.8)
    W["Conditional quintiles (GARCH)"] = W["MM on GARCH"].where((rv1 >= q80) | (rv1 <= q20), 1.0)
    W["Recovery state (RV)"] = W["MM on RV"].where(~((rv1 > q80) & (rv5 < rv63)), np.maximum(W["MM on RV"], 1.0))
    nets = {}
    for k, w in W.items():
        w = w.loc["1970-01":]; sim = simulate(w, e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index; nets[k] = sim.net_ex
    out = {}
    for p, (a, b) in PERIODS.items():
        n = len(nets["Unmanaged"].loc[a:b]); BB, BL = 4000, 6
        st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
        col = {}
        for k, z in nets.items():
            v0 = sr(z.loc[a:b])
            if k == "Unmanaged":
                col[k] = f"{v0:.2f}"; continue
            A, B = z.loc[a:b].to_numpy(), nets["Unmanaged"].loc[a:b].to_numpy()
            d0 = v0 - sr(nets["Unmanaged"].loc[a:b]); dd_ = srb(A[I]) - srb(B[I]); lo, hi = np.percentile(dd_ - dd_.mean() + d0, [2.5, 97.5])
            col[k] = f"{v0:.2f} ({d0:+.2f} [{lo:+.2f}, {hi:+.2f}])"
        out[p] = col
    print(f"\n  {gname}")
    print(pd.DataFrame(out).to_string())
