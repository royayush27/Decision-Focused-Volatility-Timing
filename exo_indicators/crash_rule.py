"""Do markets rebound after big crashes, and does a 'don't de-lever after a crash' override help vol timing?
Triggers fixed in advance, real-time:
  CRASH : last month's return below the 5th percentile of all earlier months (expanding)
  BEAR  : last month-end drawdown from the running peak worse than -20%
Override: exposure = max(w_MM, 1) in trigger months (never hold less than the unmanaged portfolio).
Facts use 1963-2025 Mkt-RF. Strategy test uses the project protocol: c frozen on 1970-1999, dev 2000-2024, kappa 5 bps."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]


def triggers(ex):
    q5 = ex.expanding(60).quantile(0.05).shift(1)                 # 5th pct of months before t-1
    crash = (ex.shift(1) < q5.shift(0)).fillna(False)              # month t-1 return below it
    wealth = (1 + ex).cumprod()
    dd = (wealth / wealth.cummax() - 1).shift(1)                   # drawdown at end of t-1
    return crash, (dd < -0.20).fillna(False)


# ---------- facts: what happens the month after a trigger (market, 1963-2025)
mk = m[m.series == "Mkt-RF"].set_index("month").sort_index()
ex, rv = mk["ret_m"], mk["rv"]
crash, bear = triggers(ex)
print("Market 1963-2025: the month AFTER a trigger")
for name, g in (("after CRASH", crash), ("in BEAR", bear), ("all other", ~(crash | bear))):
    R, Q = ex[g], rv[g]
    print(f"  {name:12s} n={g.sum():3d}  mean {R.mean()*100:+.2f}%  median {R.median()*100:+.2f}%  "
          f"share up {(R > 0).mean():.2f}  sd {R.std()*100:.2f}%  mean RV {Q.mean()*1e4:.1f}  "
          f"mean/RV (what sizing cares about) {R.mean()/Q.mean():.2f}")
print("  crash months and the next month's return:")
ev = crash[crash].index
print("   " + ", ".join(f"{ex.index[ex.index.get_loc(t)-1]} {ex.shift(1)[t]*100:+.1f}% -> {ex[t]*100:+.1f}%" for t in ev))

# ---------- strategy test, market and panel
rows, nets = [], {}
for s in SERIES:
    ms = m[m.series == s].set_index("month").sort_index().loc["1963-07":"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    v = pd.read_csv(f"dfl_diagnosis/panel/{s}_garch.csv", index_col="month")["qmle"]
    e, rf = e.loc["1970-01":], rf.loc["1970-01":]; v = v.reindex(e.index)
    cr, br = triggers(ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf_all.reindex(ms.index))
    cr, br = cr.reindex(e.index), br.reindex(e.index)
    w0 = pd.Series(exposure(v, solve_c(v.loc[:"1999-12"])), index=e.index)
    rules = {"MM": w0, "MM+crash": w0.where(~cr, np.maximum(w0, 1.0)), "MM+bear": w0.where(~br, np.maximum(w0, 1.0)),
             "unmanaged": pd.Series(1.0, index=e.index)}
    for k, w in rules.items():
        sim = simulate(w, e, rf, 5e-4); sim.index = e.index
        z = sim.net_ex.loc["2000-01":"2024-12"]; nets[(k, s)] = z.to_numpy()
        rows.append({"series": s, "rule": k, "SR": z.mean() / z.std() * np.sqrt(12),
                     "n_trig": int((cr if k == "MM+crash" else br).loc["2000-01":"2024-12"].sum()) if "+" in k else 0})
T = pd.DataFrame(rows).pivot(index="series", columns="rule", values="SR")
print("\nNet Sharpe 2000-2024 (GARCH QMLE vol forecast, c frozen on 1970-1999)")
print(T.round(3).loc[["Mkt-RF"]].to_string())
n = len(nets[("MM", "Mkt-RF")]); BB, BL = 4000, 6
rng = np.random.default_rng(0); st = rng.integers(0, n, (BB, int(np.ceil(n / BL))))
I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
for k in ("MM+crash", "MM+bear"):
    for label, sers in (("market", ["Mkt-RF"]), ("panel 16", SERIES)):
        dd = np.mean([srb(nets[(k, s)][I]) - srb(nets[("MM", s)][I]) for s in sers], axis=0)
        d0 = np.mean([T.loc[s, k] - T.loc[s, "MM"] for s in sers])
        print(f"  {k:9s} vs MM, {label:8s}: dSR {d0:+.3f} [{np.percentile(dd-dd.mean()+d0,2.5):+.3f}, "
              f"{np.percentile(dd-dd.mean()+d0,97.5):+.3f}]  better {sum(T.loc[s,k] > T.loc[s,'MM'] for s in sers)}/{len(sers)}")
trig = pd.DataFrame(rows).query("rule != 'MM' and rule != 'unmanaged'").groupby("rule").n_trig.agg(["min", "median", "max"])
print("trigger months per series in 2000-2024:\n", trig.to_string())
