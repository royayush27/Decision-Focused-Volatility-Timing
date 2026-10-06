"""Rule-design fixes for the mean-timing problem (no new data), against plain Moreira-Muir (MM).
  Xu (2024, Critical Finance Review), two switches on top of MM, both known at the end of t-1:
    delta: hold the series only if its mean return over the past 12 months is positive, otherwise cash.
    gamma: vol-manage only if the 60-month rolling regression r_t = a + b * v_t has a > 0, otherwise exposure 1
           (a <= 0 means expected returns rise about one-for-one with variance, so scaling by 1/v does not pay).
  Bongaerts, Kang, van Dijk (2020, FAJ), conditional volatility targeting: use the MM exposure only when v_t is in
    the top or bottom quintile of its expanding history (from 120 months of data), otherwise exposure 1.
Variance v_t: last month's realised variance (MM) on all 16 series; GARCH (results_1970) for the market as well.
Protocol as elsewhere: c frozen on 1970-1999 (mean exposure 1), evaluation 2000-2024, w in [0, 2], 5 bps per unit turnover.
Paired block bootstrap (6-month blocks, common across series) for the change in net Sharpe vs MM."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS

SERIES = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm",
          "Shops", "Hlth", "Utils", "Other"]
EV = ("2000-01", "2024-12")
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
mf = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")


def rolling_intercept(r, v, window=60):
    a = pd.Series(np.nan, index=r.index)
    for i in range(window, len(r)):
        y, x = r.iloc[i - window:i].to_numpy(), v.iloc[i - window:i].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 48:
            a.iloc[i] = np.polyfit(x[ok], y[ok], 1)[1]
    return a


def rules(e, v):
    """Exposure series for every rule; e = excess returns, v = variance known at the end of t-1 (same index)."""
    mm = pd.Series(exposure(v, solve_c(v.loc["1970-01":"1999-12"])), index=e.index)
    hold = (e.rolling(12).mean().shift(1) > 0)
    scale = rolling_intercept(e, v) > 0
    q20, q80 = v.expanding(120).quantile(0.2), v.expanding(120).quantile(0.8)
    extreme = (v >= q80) | (v <= q20)
    W = {"MM": mm,
         "Xu: vol-manage only if intercept > 0": mm.where(scale, 1.0),
         "Xu: cash if 12m mean < 0": mm.where(hold, 0.0),
         "Xu: both switches": mm.where(scale, 1.0).where(hold, 0.0),
         "Conditional (top/bottom quintile)": mm.where(extreme, 1.0),
         "Unmanaged": pd.Series(1.0, index=e.index)}
    return {k: w.loc["1970-01":] for k, w in W.items()}


def evaluate(W, e, rf):
    out, nets = {}, {}
    for k, w in W.items():
        sim = simulate(w, e.loc[w.index], rf.loc[w.index], 5e-4); sim.index = w.index
        z = sim.net_ex.loc[EV[0]:EV[1]]; nets[k] = z.to_numpy()
        x = w.loc[EV[0]:EV[1]]; R = e.loc[EV[0]:EV[1]]; xn = x / x.mean()
        out[k] = {"SR_net": z.mean() / z.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(),
                  "VT": R.std() / (xn * R).std(), "avg_w": x.mean(), "turnover": x.diff().abs().mean()}
    return pd.DataFrame(out).T, nets


rng = np.random.default_rng(0)
def boot_idx(n, BB=4000, BL=6):
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL))))
    return ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)


def ci(dd, d0):
    lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
    return f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}] p={np.mean(np.abs(dd - dd.mean()) >= abs(d0)):.3f}"


tables, nets = {}, {}
for s in SERIES:
    ms = m[m["series"] == s].set_index("month").sort_index().loc["1963-07":"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    tables[s], nets[s] = evaluate(rules(e, ms["rv"].shift(1)), e, rf)
    if s == "Mkt-RF":
        e_g = e.loc["1970-01":]; v_g = mf["garch_rv_forecast"].reindex(e_g.index)
        tables["Mkt-RF GARCH"], nets["Mkt-RF GARCH"] = evaluate(rules(e_g, v_g), e_g, rf.loc[e_g.index])

pd.set_option("display.width", 200)
I = boot_idx(len(nets["Mkt-RF"]["MM"]))
for lab in ("Mkt-RF", "Mkt-RF GARCH"):
    print(f"\n{lab}, {EV[0]} to {EV[1]}, net of 5 bps")
    print(tables[lab].round(3).to_string())
    for k in list(tables[lab].index)[1:]:
        d0 = tables[lab].loc[k, "SR_net"] - tables[lab].loc["MM", "SR_net"]
        print(f"  {k:36s} - MM: dSR {ci(srb(nets[lab][k][I]) - srb(nets[lab]['MM'][I]), d0)}")

P = pd.DataFrame({s: tables[s]["SR_net"] for s in SERIES}).T
print(f"\n16-series panel (variance = last month's RV), mean net SR")
print(P.mean().round(3).to_string())
for k in list(P.columns)[1:]:
    dd = np.mean([srb(nets[s][k][I]) - srb(nets[s]["MM"][I]) for s in SERIES], axis=0)
    d0 = (P[k] - P["MM"]).mean()
    print(f"  {k:36s} - MM: mean dSR {ci(dd, d0)}  better {int((P[k] > P['MM']).sum())}/16")
