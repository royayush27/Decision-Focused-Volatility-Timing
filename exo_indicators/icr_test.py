"""Intermediary capital ratio (He, Kelly, Manela 2017) as an expected-return signal: when primary dealers are short of
capital, the risk premium should be high, so a vol-managed portfolio should not cut exposure as much.
Data: exo_indicators/hkm_capital_ratio_monthly.csv, from He_Kelly_Manela_Factors_monthly_250627.csv
(zhiguohe.net, column intermediary_capital_ratio, 1970-01 to 2025-05; the source lists 2025-01 twice, the later row is kept).
Timing: the ratio uses dealers' market equity (month end) and book debt (quarterly, reported with a delay), so the main
specification uses the value for month t-2 at the end of t-1; t-1 (optimistic) and t-4 (safe for quarterly debt) are robustness.
Two specifications, fixed in advance:
  A. mu rule: w = c * max(mu_hat, 0) / v_hat, mu_hat from an expanding OLS of next-month excess return on log ICR.
     Benchmark: the same rule with the expanding historical mean (= Moreira-Muir up to c).
  B. MM override: Moreira-Muir exposure, but never below 1 when ICR is in the bottom quintile of its own expanding history.
     Benchmark: MM.
Market: v_hat = GARCH (results_1970) and last month's RV. Panel (16 series): last month's RV, the market-level ICR signal.
Estimation from 1970, OOS forecasts from 1980, c calibrated 1980-1999, evaluation 2000-2024, 5 bps, w in [0, 2]."""
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
EST, OOS, CAL, EV = "1970-01", "1980-01", ("1980-01", "1999-12"), ("2000-01", "2024-12")
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
rf_all = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
mf = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")
icr = pd.read_csv("exo_indicators/hkm_capital_ratio_monthly.csv", index_col="month")["icr"]
assert icr.index.is_unique and icr.index[0] == "1970-01"


def signal(lag, index):
    return np.log(icr).shift(lag - 1).reindex(index)      # value of month t-lag, used for month t (shift 1 = month t-1)


def mu_oos(e, x):
    mu, hist = pd.Series(np.nan, index=e.index), pd.Series(np.nan, index=e.index)
    for t in e.loc[OOS:"2025-12"].index:
        tr = e.loc[EST:t].index[:-1]
        hist[t] = e.loc[tr].mean()
        ok = x.loc[tr].notna() & e.loc[tr].notna()
        if ok.sum() >= 60 and pd.notna(x.loc[t]):
            b, a = np.polyfit(x.loc[tr][ok], e.loc[tr][ok], 1)
            mu[t] = a + b * x.loc[t]
    return mu, hist


def mu_rule(mu, v):
    sig = (np.maximum(mu, 0) / v).where(lambda z: z > 0, 1e-12)
    return pd.Series(np.clip(solve_c(1 / sig.loc[CAL[0]:CAL[1]].dropna()) * sig, 0, 2), index=v.index).fillna(1.0)


def override(v, x):
    mm = pd.Series(exposure(v, solve_c(v.loc[CAL[0]:CAL[1]])), index=v.index)
    low = x <= x.expanding(120).quantile(0.2)              # x = log ICR, expanding quintile uses data up to that month
    return mm, mm.where(~low, np.maximum(mm, 1.0)), low


def stats(w, e, rf):
    sim = simulate(w, e, rf, 5e-4); sim.index = e.index
    z = sim.net_ex.loc[EV[0]:EV[1]]; x = w.loc[EV[0]:EV[1]]; R = e.loc[EV[0]:EV[1]]; xn = x / x.mean()
    return {"SR_net": z.mean() / z.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(), "VT": R.std() / (xn * R).std(),
            "avg_w": x.mean(), "share_cash": (x < 1e-6).mean()}, z.to_numpy()


rng = np.random.default_rng(0)
n_ev = len(pd.period_range(EV[0], EV[1], freq="M")); BB, BL = 4000, 6
_st = rng.integers(0, n_ev, (BB, int(np.ceil(n_ev / BL)))); I = ((_st[:, :, None] + np.arange(BL)) % n_ev).reshape(BB, -1)[:, :n_ev]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
def ci(dd, d0):
    lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
    return f"{d0:+.3f} [{lo:+.3f}, {hi:+.3f}] p={np.mean(np.abs(dd - dd.mean()) >= abs(d0)):.3f}"


def series_data(s):
    ms = m[m["series"] == s].set_index("month").sort_index().loc[EST:"2025-12"]
    rf = rf_all.reindex(ms.index); e = ms["ret_m"] if s in FACTORS else ms["ret_m"] - rf
    rv_all = m[m["series"] == s].set_index("month").sort_index()["rv"].shift(1).reindex(ms.index)
    return e, rf, rv_all


pd.set_option("display.width", 200)
e, rf, rv1 = series_data("Mkt-RF")
print(f"Market (Mkt-RF), evaluation {EV[0]} to {EV[1]}; ICR lag 2 unless stated")
for vlab, v in (("GARCH", mf["garch_rv_forecast"].reindex(e.index)), ("RV", rv1)):
    rows, nets = {}, {}
    for lag in (2, 1, 4):
        x = signal(lag, e.index); mu, hist = mu_oos(e, x)
        R = e.loc[EV[0]:EV[1]]
        r2 = 1 - ((R - mu.loc[R.index]) ** 2).sum() / ((R - hist.loc[R.index]) ** 2).sum()
        if lag == 2:
            rows["A benchmark: hist mean"], nets["A0"] = stats(mu_rule(hist, v), e, rf)
        rows[f"A: ICR mu rule, lag {lag} (OOS R2 {100 * r2:+.2f}%)"], nets[f"A{lag}"] = stats(mu_rule(mu, v), e, rf)
        mm, ov, low = override(v, x)
        if lag == 2:
            rows["B benchmark: MM"], nets["B0"] = stats(mm, e, rf)
        rows[f"B: MM, no cut when ICR low, lag {lag} ({100 * low.loc[EV[0]:EV[1]].mean():.0f}% of months)"], nets[f"B{lag}"] = stats(ov, e, rf)
    print(f"\n  variance forecast: {vlab}")
    print(pd.DataFrame(rows).T.round(3).to_string())
    for k, b in (("A2", "A0"), ("A1", "A0"), ("A4", "A0"), ("B2", "B0"), ("B1", "B0"), ("B4", "B0")):
        d0 = (nets[k].mean() / nets[k].std(ddof=1) - nets[b].mean() / nets[b].std(ddof=1)) * np.sqrt(12)
        print(f"    {k} - {b}: dSR {ci(srb(nets[k][I]) - srb(nets[b][I]), d0)}")

print("\n16-series panel, variance = last month's RV, market ICR (lag 2)")
res = {k: [] for k in ("A", "A0", "B", "B0")}; NET = {k: [] for k in res}
for s in SERIES:
    e, rf, v = series_data(s)
    x = signal(2, e.index); mu, hist = mu_oos(e, x)
    for k, w in (("A", mu_rule(mu, v)), ("A0", mu_rule(hist, v))):
        st, z = stats(w, e, rf); res[k].append(st["SR_net"]); NET[k].append(z)
    mm, ov, _ = override(v, x)
    for k, w in (("B", ov), ("B0", mm)):
        st, z = stats(w, e, rf); res[k].append(st["SR_net"]); NET[k].append(z)
for k, b, lab in (("A", "A0", "ICR mu rule vs hist mean"), ("B", "B0", "MM no-cut-when-ICR-low vs MM")):
    dd = np.mean([srb(NET[k][i][I]) - srb(NET[b][i][I]) for i in range(len(SERIES))], axis=0)
    d0 = np.mean(np.array(res[k]) - np.array(res[b]))
    print(f"  {lab:32s}: mean SR {np.mean(res[k]):.3f} vs {np.mean(res[b]):.3f}, mean dSR {ci(dd, d0)}, "
          f"better {int(sum(a > c for a, c in zip(res[k], res[b])))}/16")
