"""International out-of-sample test of the US result: EW Mkt + profitability + investment + momentum, vol-managed with MM on GARCH.
Data: JKP Global Factor Data (jkpfactors.com), capped value-weighted (vw_cap), USD excess returns, daily and monthly,
saved in data/jkp/<region>_{daily,monthly}.csv. Closest JKP analogues of the US factors:
  Mkt-RF -> mkt,  RMW -> ope_be (operating profits / book equity),  CMA -> at_gr1 (asset growth, signed low-minus-high),
  Mom -> ret_12_1 (price momentum t-12 to t-1). JKP factors are tercile long-short within each country, so they are
  not identical to Fama-French factors. Nothing below was tuned on these data: same rule and settings as mix_curves_1998.py.
Rule: real-time GARCH(1,1) on the mix's daily returns (garch_tools.py), first refit in January of first_year
(2 full years of daily data after the mix starts, not before 1990); c frozen on first_year..1997 so mean exposure = 1; w in [0, 2];
5 bps per unit of exposure change. Evaluation 1998-2025 and sub-periods. US T-bill (Ken French RF) only for wealth/drawdown.
Inference: paired 6-month block bootstrap of dSR = SR(MM) - SR(unmanaged); the pooled test uses blocks common to the
seven non-overlapping regions (six countries + emerging; world ex US is reported separately since it contains them).
Writes main_eval/results/intl_mix_1998.csv."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import numpy as np
import pandas as pd

from backtest import simulate
from calibrate_c import exposure, solve_c
from garch_tools import garch_monthly

REGIONS = ["world_ex_us", "jpn", "gbr", "deu", "fra", "can", "aus", "emerging"]
POOL = [r for r in REGIONS if r != "world_ex_us"]
MIXES = {"Mkt + Prof + Inv + Mom": ["mkt", "ope_be", "at_gr1", "ret_12_1"],
         "Prof + Inv + Mom": ["ope_be", "at_gr1", "ret_12_1"],
         "Mkt + Mom": ["mkt", "ret_12_1"],
         "Market alone": ["mkt"],
         "Mkt + Prof + Inv (no Mom)": ["mkt", "ope_be", "at_gr1"],
         "Mom alone": ["ret_12_1"]}
PERIODS = {"1998-2025": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"),
           "2016-2025": ("2016-01", "2025-12"), "2021-2025": ("2021-01", "2025-12")}
OUT = _P("main_eval/results"); OUT.mkdir(parents=True, exist_ok=True)
rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
rng = np.random.default_rng(0); BB, BL = 4000, 6


def load(reg, freq):
    x = pd.read_csv(f"data/jkp/{reg}_{freq}.csv", dtype={"date": str})
    x.index = pd.to_datetime(x.pop("date"), format="%Y%m%d")
    return x


def stats(z, r):
    wl = np.r_[1.0, (1 + z + r).cumprod().to_numpy()]
    dn = np.sqrt(np.mean(np.minimum(z, 0) ** 2))
    return {"Sharpe": z.mean() / z.std() * np.sqrt(12), "Sortino": z.mean() * 12 / (dn * np.sqrt(12)),
            "Vol_%": 100 * z.std() * np.sqrt(12), "MaxDD_%": 100 * (wl / np.maximum.accumulate(wl) - 1).min()}


def sr(M):    # annualised Sharpe of each row
    return M.mean(-1) / M.std(-1, ddof=1) * np.sqrt(12)


nets, rows, info = {}, [], []
for reg in REGIONS:
    D, M = load(reg, "daily"), load(reg, "monthly")
    M.index = M.index.to_period("M").astype(str)
    for mix, fs in MIXES.items():
        e = M[fs].dropna().mean(axis=1)
        start = e.index[0]
        d = D[fs].loc[D[fs].dropna().index[0]:].fillna(0.0).mean(axis=1)   # missing day of one factor = 0 return that day
        fy = max(1990, int(start[:4]) + 2 + (start[5:] != "01"))
        v = garch_monthly(f"intl_{reg}_" + "_".join(fs), d, first_year=fy).reindex(e.index).dropna()
        e = e.loc[v.index]
        cal = (f"{fy}-01", "1997-12")
        w = pd.Series(exposure(v, solve_c(v.loc[cal[0]:cal[1]])), index=v.index)
        sim = simulate(w, e, rf.reindex(e.index), 5e-4); sim.index = e.index
        nets[(reg, mix, "MM-GARCH")], nets[(reg, mix, "Unmanaged")] = sim.net_ex, e
        info.append({"region": reg, "mix": mix, "data from": start, "cal window": f"{cal[0][:4]}-1997",
                     "mean w 1998+": round(w.loc["1998-01":].mean(), 2)})
        for p, (a, b) in {"cal": cal, **PERIODS}.items():
            for ver in ("MM-GARCH", "Unmanaged"):
                rows.append({"region": reg, "mix": mix, "period": p, "version": ver,
                             **stats(nets[(reg, mix, ver)].loc[a:b], rf.loc[a:b])})
T = pd.DataFrame(rows); T.to_csv(OUT / "intl_mix_1998.csv", index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(info).query("mix == 'Mkt + Prof + Inv + Mom'").to_string(index=False))


def gain_ci(reg, mix, a, b):
    m = nets[(reg, mix, "MM-GARCH")].loc[a:b].to_numpy(); u = nets[(reg, mix, "Unmanaged")].loc[a:b].to_numpy(); n = len(m)
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    g = sr(m[I]) - sr(u[I])
    return sr(m) - sr(u), *np.percentile(g, [2.5, 97.5])


for mix in MIXES:
    print(f"\n================ {mix}")
    for p, (a, b) in {"in-sample cal": ("1990-01", "1997-12"), **PERIODS}.items():
        P = T[(T.mix == mix) & (T.period == ("cal" if p == "in-sample cal" else p))].pivot(index="region", columns="version", values=["Sharpe", "Sortino", "MaxDD_%"]).loc[REGIONS]
        if p != "in-sample cal":
            P[("gain", "dSR")], P[("gain", "lo")], P[("gain", "hi")] = zip(*[gain_ci(r, mix, a, b) for r in REGIONS])
        print(f"\n--- {p}" + (" (each region's own calibration window, in-sample for c)" if p == "in-sample cal" else ""))
        print(P.round(2).to_string())
        if p == "in-sample cal":
            continue
        # pooled: mean dSR over the 7 non-overlapping regions, common blocks
        Mm = np.stack([nets[(r, mix, "MM-GARCH")].loc[a:b].to_numpy() for r in POOL])
        Mu = np.stack([nets[(r, mix, "Unmanaged")].loc[a:b].to_numpy() for r in POOL]); n = Mm.shape[1]
        st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
        g = (sr(Mm[:, I]) - sr(Mu[:, I])).mean(0); g0 = (sr(Mm) - sr(Mu))
        lo, hi = np.percentile(g, [2.5, 97.5])
        print(f"pooled 7 regions: mean dSR {g0.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], regions better {int((g0 > 0).sum())}/7")
