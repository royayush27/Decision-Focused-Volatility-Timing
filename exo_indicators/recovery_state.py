"""Out-of-sample test of the 'recovery phase' idea from vix_term.py.
Idea (found in 2008-2024 VIX data, so NOT tested there): months after high volatility in which fear has passed have high
reward per unit of risk, but Moreira-Muir (MM) is still mostly out. Here the state is defined with realised volatility only,
so it can be tested on 1965-2007 (and on the 16-series panel), which the idea was not built on.
Specification, fixed before running (decision at the end of month t-1, daily data up to that day):
  RV21 = last month's realised variance (the MM input); RV5 = sum of the last 5 squared daily returns * 21/5;
  RV63 = sum of the last 63 squared daily returns * 21/63.
  high     = RV21 above the 80th percentile of its own expanding history (at least 120 months)
  recovery = high and RV5 <  RV63   (volatility has peaked and is falling: the realised analogue of a normal VIX curve)
  panic    = high and RV5 >= RV63
  H1: next-month excess return per unit of variance is higher in recovery than in panic months.
  Rule: MM, but exposure at least 1 in recovery months. Benchmark: MM. w in [0, 2], 5 bps per unit turnover.
  c calibrated 1965-1974, evaluation 1975-2007 (main, untouched). 2008-2024 is reported as an in-sample check only."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis"):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
CAL = ("1965-01", "1974-12")
PERIODS = {"1975-2007 (out of sample)": ("1975-01", "2007-12"), "2008-2024 (in-sample check)": ("2008-01", "2024-12")}
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
d = pd.read_csv("data/ff_daily_garchdata_long.csv")
d["date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")


def states(s, months):
    r = d[d["series"] == s].set_index("date")["ret"].sort_index()
    eom = r.groupby(r.index.to_period("M")).apply(lambda z: z.index[-1])          # last trading day of each month
    rv5 = ((r ** 2).rolling(5).sum() * 21 / 5).loc[eom.values]
    rv63 = ((r ** 2).rolling(63).sum() * 21 / 63).loc[eom.values]
    out = pd.DataFrame({"rv5": rv5.to_numpy(), "rv63": rv63.to_numpy()}, index=eom.index.astype(str))
    return out.shift(1).reindex(months)                                            # end of t-1 values, for month t


rows, nets, desc = [], {}, []
for s in SERIES:
    ms = m[m["series"] == s].set_index("month").sort_index().loc["1963-07":"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    rv21 = ms["rv"].shift(1)
    st = states(s, ms.index)
    high = rv21 > rv21.expanding(120).quantile(0.8)
    rec = high & (st["rv5"] < st["rv63"]); pan = high & (st["rv5"] >= st["rv63"])
    sl = slice("1965-01", "2025-12"); e, rf, rv21, rec, pan = (z.loc[sl] for z in (e, rf, rv21, rec, pan))
    mm = pd.Series(exposure(rv21, solve_c(rv21.loc[CAL[0]:CAL[1]])), index=e.index)
    W = {"MM": mm, "MM, at least 1 in recovery": mm.where(~rec, np.maximum(mm, 1.0))}
    for plab, (a, b) in PERIODS.items():
        R = e.loc[a:b]; V = ms["rv"].loc[a:b]
        for lab, msk in (("recovery", rec), ("panic", pan), ("not high", ~(rec | pan))):
            k = msk.loc[a:b]
            desc.append({"series": s, "period": plab, "state": lab, "months": int(k.sum()), "mean_ret": R[k].mean(),
                         "ret_per_var": R[k].mean() / V[k].mean() if k.sum() else np.nan, "mean_w_MM": mm.loc[a:b][k].mean()})
        for k, w in W.items():
            sim = simulate(w, e, rf, 5e-4); sim.index = e.index
            z = sim.net_ex.loc[a:b]; nets[(plab, k, s)] = z.to_numpy()
            x = w.loc[a:b]; xn = x / x.mean()
            rows.append({"series": s, "period": plab, "rule": k, "SR": z.mean() / z.std() * np.sqrt(12),
                         "MT": (xn * R).mean() / R.mean(), "VT": R.std() / (xn * R).std()})

D, T = pd.DataFrame(desc), pd.DataFrame(rows)
pd.set_option("display.width", 200)
rng = np.random.default_rng(0)
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
for plab, (a, b) in PERIODS.items():
    print(f"\n==================== {plab}")
    mk = D[(D.series == "Mkt-RF") & (D.period == plab)].set_index("state")
    print("Market: next-month excess return by state at the end of the previous month")
    print((mk[["months", "mean_ret", "ret_per_var", "mean_w_MM"]] * [1, 100, 1, 1]).rename(columns={"mean_ret": "mean_ret_%"}).round(3).to_string())
    e_mk = m[m.series == "Mkt-RF"].set_index("month")["ret_m"].loc[a:b]
    print("Market rules:")
    print(T[(T.series == "Mkt-RF") & (T.period == plab)].set_index("rule")[["SR", "MT", "VT"]].round(3).to_string())
    n = len(nets[(plab, "MM", "Mkt-RF")]); BB, BL = 4000, 6
    stt = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((stt[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    P = T[T.period == plab].pivot(index="series", columns="rule", values="SR")
    for lab, sers in (("market", ["Mkt-RF"]), ("panel ", SERIES)):
        dd = np.mean([srb(nets[(plab, "MM, at least 1 in recovery", s)][I]) - srb(nets[(plab, "MM", s)][I]) for s in sers], axis=0)
        d0 = (P.loc[sers, "MM, at least 1 in recovery"] - P.loc[sers, "MM"]).mean()
        lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5]); p = np.mean(np.abs(dd - dd.mean()) >= abs(d0))
        nb = int((P.loc[sers, "MM, at least 1 in recovery"] > P.loc[sers, "MM"]).sum())
        print(f"  dSR rule - MM, {lab}: {d0:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}  better {nb}/{len(sers)}")
    pv = D[D.period == plab].pivot(index="series", columns="state", values="ret_per_var")
    print(f"  H1 across 16 series: reward per unit variance recovery > panic in {int((pv['recovery'] > pv['panic']).sum())}/16; "
          f"median recovery {pv['recovery'].median():.2f}, panic {pv['panic'].median():.2f}, not high {pv['not high'].median():.2f}")
