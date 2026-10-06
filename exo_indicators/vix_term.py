"""VIX term structure as a panic-peak / recovery signal (market, monthly). inverted(t) = VIX > VIX3M at the close of the
last trading day of month t (CBOE 30-day vs 3-month implied vol; data from Dec 2007). Known at the end of t, used for t+1.
Data: exo_indicators/vix_term_structure_monthly.csv (FRED VIXCLS and VXVCLS, month-end values on the same day and the share of
days in the month with VIX > VIX3M; built in the browser from the FRED CSVs, checksums verified).
Part 1, descriptive: next-month market excess return and realised variance by state, with and without a high-RV filter.
Part 2, rules against Moreira-Muir (MM on last month's RV, c calibrated 1990-2007, evaluation 2008-02 to 2024-12, 5 bps, w in [0, 2]):
  T1 'no cut while inverted': exposure at least 1 in the month after an inverted month-end.
  T2 'back in after the panic': exposure at least 1 in the month after the curve normalises (inverted at t-1, not at t).
  T3 mu rule w = c * max(mu_hat, 0) / RV with mu_hat from an expanding OLS on log(VIX / VIX3M); OOS from 2013 (60 months of data),
     c calibrated 2013-2015, evaluation 2016-2024 (short; indicative only)."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c

mf = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")
rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"].reindex(mf.index)
ts = pd.read_csv("exo_indicators/vix_term_structure_monthly.csv", index_col="month")
ex, rv = mf["ret_m"], mf["rv"]
ratio = (ts["vix"] / ts["vix3m"]).reindex(mf.index)
inv = (ratio > 1).astype(float).where(ratio.notna())
inv_l1, inv_l2 = inv.shift(1), inv.shift(2)                  # state known at the end of t-1 (and t-2), for month t
rv_l1 = rv.shift(1)
hi_rv = rv_l1 > rv_l1.expanding(120).quantile(0.8)

idx = mf.loc["2008-02":"2024-12"].index
print("Part 1. Next-month market excess return by state at the end of the previous month, 2008-02 to 2024-12")
def row(mask, lab):
    z = ex.loc[idx][mask.loc[idx].fillna(False).astype(bool)]
    v = rv.loc[idx][mask.loc[idx].fillna(False).astype(bool)]
    return {"state": lab, "months": len(z), "mean ret %/mo": 100 * z.mean(), "t": z.mean() / z.std() * np.sqrt(len(z)) if len(z) > 2 else np.nan,
            "mean RV (ann. vol %)": 100 * np.sqrt(12 * v.mean()), "ret / RV": z.mean() / v.mean()}
all_ = pd.Series(True, index=mf.index)
T = pd.DataFrame([row(all_, "all months"), row(inv_l1 == 1, "inverted"), row(inv_l1 == 0, "not inverted"),
                  row((inv_l2 == 1) & (inv_l1 == 0), "just normalised"), row(hi_rv, "top-20% RV"),
                  row(hi_rv & (inv_l1 == 1), "top-20% RV and inverted"), row(hi_rv & (inv_l1 == 0), "top-20% RV, not inverted")])
pd.set_option("display.width", 200)
print(T.set_index("state").round(3).to_string())
print("  ret / RV is the reward per unit of variance: MM assumes it is the same in every state.")

c = solve_c(rv_l1.loc["1990-01":"2007-12"])
mm = pd.Series(exposure(rv_l1, c), index=mf.index)
W = {"MM": mm,
     "T1 no cut while inverted": mm.where(inv_l1 != 1, np.maximum(mm, 1.0)),
     "T2 back in after the panic": mm.where(~((inv_l2 == 1) & (inv_l1 == 0)), np.maximum(mm, 1.0)),
     "Unmanaged": pd.Series(1.0, index=mf.index)}


def evaluate(W, a, b):
    out, nets = {}, {}
    for k, w in W.items():
        sim = simulate(w.fillna(1.0), ex, rf, 5e-4); sim.index = mf.index
        z = sim.net_ex.loc[a:b]; x = w.loc[a:b]; R = ex.loc[a:b]; xn = x / x.mean(); wl = (1 + z).cumprod()
        nets[k] = z.to_numpy()
        out[k] = {"SR_net": z.mean() / z.std() * np.sqrt(12), "MT": (xn * R).mean() / R.mean(), "VT": R.std() / (xn * R).std(),
                  "MaxDD": (wl / wl.cummax() - 1).min(), "avg_w": x.mean(), "months overridden": int((w.loc[a:b] != mm.loc[a:b]).sum())}
    return pd.DataFrame(out).T, nets


rng = np.random.default_rng(0)
def report(W, a, b, label):
    Tb, nets = evaluate(W, a, b)
    print(f"\n{label}, {a} to {b}, net of 5 bps")
    print(Tb.round(3).to_string())
    n = len(nets["MM"]); BB, BL = 4000, 6
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
    for k in Tb.index[1:]:
        d0 = Tb.loc[k, "SR_net"] - Tb.loc["MM", "SR_net"]; dd = srb(nets[k][I]) - srb(nets["MM"][I])
        lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
        print(f"  {k:28s} - MM: dSR {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")


report(W, "2008-02", "2024-12", "Part 2. Override rules")

x = np.log(ratio).shift(1)                                    # known at the end of t-1
mu, hist = pd.Series(np.nan, index=mf.index), pd.Series(np.nan, index=mf.index)
for t in mf.loc["2013-01":"2025-12"].index:
    tr = mf.loc["2008-01":t].index[:-1]; ok = x.loc[tr].notna()
    hist[t] = ex.loc[tr].mean()
    b, a = np.polyfit(x.loc[tr][ok], ex.loc[tr][ok], 1); mu[t] = a + b * x.loc[t]


def mu_rule(m):
    sig = (np.maximum(m, 0) / rv_l1).where(lambda z: z > 0, 1e-12)
    return pd.Series(np.clip(solve_c(1 / sig.loc["2013-01":"2015-12"].dropna()) * sig, 0, 2), index=mf.index)


R = ex.loc["2016-01":"2024-12"]
r2 = 1 - ((R - mu.loc[R.index]) ** 2).sum() / ((R - hist.loc[R.index]) ** 2).sum()
W3 = {"MM": mu_rule(hist), "T3 mu rule on log(VIX/VIX3M)": mu_rule(mu)}
mm_hist = W3["MM"]
mm = mm_hist                                                   # 'months overridden' compares with this benchmark
report(W3, "2016-01", "2024-12", f"Part 3. mu rule (OOS R2 of returns {100 * r2:+.2f}%); benchmark = same rule with the historical mean")
