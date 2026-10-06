"""Does re-entering faster fix the mean-timing problem? Same Moreira-Muir idea (exposure = c / variance estimate), but
rebalanced monthly, weekly or daily, with a 21-day or a 5-day realised-variance window. Market only (Mkt-RF daily).
Timing: the exposure for day d uses returns up to day d-1 and is set on rebalance days only (monthly: first trading day of
the month; weekly: first trading day of the week; daily: every day), then held. Costs: kappa per unit of exposure change.
c is chosen so that mean daily exposure over 1970-1999 is 1 (w in [0, 2]); evaluation 2000-2024 on monthly compounded
net excess returns. 'Monthly, 21-day' is the Moreira-Muir benchmark. Daily MT and VT: SR_daily ratio = MT x VT.
Rebound check: average exposure over the 21 trading days after the 10 worst days of 2000-2024, and in April 2020 / Mar-May 2009."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from calibrate_c import solve_c

d = pd.read_csv("data/ff_daily_garchdata_long.csv")
d = d[d["series"] == "Mkt-RF"].copy()
d["date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
r = d.set_index("date")["ret"].sort_index().loc["1965-01-01":"2025-12-31"]
EV = ("2000-01-01", "2024-12-31")

var = {"21-day": (r ** 2).rolling(21).sum().shift(1),                      # known at the close of d-1, monthly units
       "5-day": (r ** 2).rolling(5).sum().shift(1) * 21 / 5}
idx = r.index
reb = {"monthly": pd.Series(idx.to_period("M"), index=idx).ne(pd.Series(idx.to_period("M"), index=idx).shift()),
       "weekly": pd.Series(idx.to_period("W"), index=idx).ne(pd.Series(idx.to_period("W"), index=idx).shift()),
       "daily": pd.Series(True, index=idx)}


def exposure(v, on):
    target = v.where(on).ffill()                      # variance estimate frozen between rebalance days
    cal = target.loc["1970-01-01":"1999-12-31"]
    c = solve_c(cal.to_numpy())
    return np.clip(c / target, 0, 2).loc["1970-01-01":]


def monthly(x):
    return (1 + x).groupby(x.index.to_period("M")).prod() - 1


worst = r.loc[EV[0]:EV[1]].nsmallest(10).index
after = sorted({i for t in worst for i in range(idx.get_loc(t) + 1, min(idx.get_loc(t) + 22, len(idx)))})
rows, nets = {}, {}
for kappa in (5e-4, 1e-3):
    for f in ("monthly", "weekly", "daily"):
        for wlab, v in var.items():
            w = exposure(v, reb[f])
            rr = r.loc[w.index]
            gross = w * rr
            net = gross - kappa * w.diff().abs().fillna(0)
            z = monthly(net.loc[EV[0]:EV[1]])
            key = (f"{f}, {wlab}", f"{kappa * 1e4:.0f} bps")
            nets[key] = z.to_numpy()
            we, re_ = w.loc[EV[0]:EV[1]], rr.loc[EV[0]:EV[1]]; xn = we / we.mean()
            wealth = (1 + z).cumprod()
            rows[key] = {"SR_net": z.mean() / z.std() * np.sqrt(12),
                         "MT_daily": (xn * re_).mean() / re_.mean(), "VT_daily": re_.std() / (xn * re_).std(),
                         "MaxDD": (wealth / wealth.cummax() - 1).min(), "avg_w": we.mean(),
                         "turnover_per_yr": we.diff().abs().sum() / 25,
                         "w after 10 worst days": w.iloc[after].mean(),
                         "w Apr 2020": w.loc["2020-04"].mean(), "w Mar-May 2009": w.loc["2009-03":"2009-05"].mean()}
u = monthly(r.loc[EV[0]:EV[1]])
pd.set_option("display.width", 220)
T = pd.DataFrame(rows).T
print(f"Market, {EV[0][:4]}-{EV[1][:4]}; unmanaged SR {u.mean() / u.std() * np.sqrt(12):.3f}, MaxDD {((1 + u).cumprod() / (1 + u).cumprod().cummax() - 1).min():.3f}")
print(T.round(3).to_string())

rng = np.random.default_rng(0); n = len(u); BB, BL = 4000, 6
st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
srb = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
print("\nNet dSR vs Moreira-Muir (monthly, 21-day), same cost level, 95% block-bootstrap CI")
for (lab, k) in rows:
    base = ("monthly, 21-day", k)
    if (lab, k) == base:
        continue
    a, b = nets[(lab, k)], nets[base]
    d0 = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * np.sqrt(12)
    dd = srb(a[I]) - srb(b[I]); lo, hi = np.percentile(dd - dd.mean() + d0, [2.5, 97.5])
    print(f"  {lab:18s} {k:7s} {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]")
