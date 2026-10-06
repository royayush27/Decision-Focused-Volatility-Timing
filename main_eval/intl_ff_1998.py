"""International check on Ken French's regional factors (same definitions as the US factors), plus leg-wise timing.
Data: Ken French Data Library, regional 5 factors + momentum (WML), daily and monthly, USD, July 1990 - August 2026,
saved in data/ff_intl/<region>_{daily,monthly}.csv (downloaded 6 Oct 2026). The US row uses the project's US files.
Regions: Developed ex US (headline), Europe, Japan, Asia Pacific ex Japan, North America, Developed, and US.
Pool = Europe + Japan + Asia Pacific ex Japan (the three non-overlapping, non-US regions).

Two ways to time a mix, same rule everywhere (MM on GARCH, w = c / v_hat, w in [0, 2], 5 bps per unit of turnover):
  "whole mix": one GARCH on the mix's daily returns, one exposure for the whole mix (as in mix_curves_1998.py, intl_mix_1998.py).
  "legs, market unmanaged": the market leg is held at weight 1; every other leg is vol-managed on its own GARCH, then EW.
  "legs, all managed": every leg vol-managed on its own GARCH, then EW.
The leg-wise designs were chosen AFTER seeing intl_mix_1998.py (the gain was momentum, the loss was the market): report them as
a post-hoc design, not as an out-of-sample discovery. The whole-mix rows on French data are a genuine second-data-source replication.
Real time: GARCH re-estimated every January from 1993 (international; data start Nov 1990) or 1970 (US); c frozen on 1993-1997
(international) or 1970-1997 (US). Evaluation 1998-2025 and sub-periods. Writes main_eval/results/intl_ff_1998.csv."""
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
from garch_tools import daily, garch_monthly

REGIONS = ["Developed_ex_US", "Europe", "Japan", "Asia_Pacific_ex_Japan", "North_America", "Developed", "US"]
POOL = ["Europe", "Japan", "Asia_Pacific_ex_Japan"]
MIXES = {"Mkt + RMW + CMA + Mom": ["Mkt-RF", "RMW", "CMA", "Mom"],
         "RMW + CMA + Mom": ["RMW", "CMA", "Mom"],
         "Mkt + Mom": ["Mkt-RF", "Mom"],
         "Mkt + RMW + CMA (no Mom)": ["Mkt-RF", "RMW", "CMA"],
         "Market alone": ["Mkt-RF"],
         "Mom alone": ["Mom"],
         "All six": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]}
PERIODS = {"1998-2025": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"),
           "2016-2025": ("2016-01", "2025-12"), "2021-2025": ("2021-01", "2025-12")}
OUT = _P("main_eval/results"); OUT.mkdir(parents=True, exist_ok=True)
rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"]
rng = np.random.default_rng(0); BB, BL = 4000, 6
FACS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def load(reg):
    if reg == "US":
        m = pd.read_csv("data/ff_monthly_baselinedata.csv")
        M = pd.DataFrame({f: m[m["series"] == f].set_index("month")["ret_m"] for f in FACS}).sort_index()
        d = daily(); D = pd.DataFrame({f: d[d["series"] == f].set_index("date")["ret"] for f in FACS}).sort_index()
        return D, M, 1970, ("1970-01", "1997-12")
    D = pd.read_csv(f"data/ff_intl/{reg}_daily.csv", dtype={"date": str}); D.index = pd.to_datetime(D.pop("date"), format="%Y%m%d")
    M = pd.read_csv(f"data/ff_intl/{reg}_monthly.csv", dtype={"date": str}); M.index = M.pop("date").map(lambda s: f"{s[:4]}-{s[4:6]}")
    return D[FACS], M[FACS], 1993, ("1993-01", "1997-12")


def managed(reg, tag, e, d, fy, cal):
    """Net excess return of e vol-managed on a GARCH of the daily series d (MM on GARCH)."""
    cache = (f"mix_" if reg == "US" else f"ffi_{reg}_") + tag
    v = garch_monthly(cache, d, first_year=fy).reindex(e.index).dropna()
    e = e.loc[v.index]
    w = pd.Series(exposure(v, solve_c(v.loc[cal[0]:cal[1]])), index=v.index)
    sim = simulate(w, e, rf.reindex(e.index), 5e-4); sim.index = e.index
    return sim.net_ex


def stats(z, r):
    wl = np.r_[1.0, (1 + z + r).cumprod().to_numpy()]
    dn = np.sqrt(np.mean(np.minimum(z, 0) ** 2))
    return {"Sharpe": z.mean() / z.std() * np.sqrt(12), "Sortino": z.mean() * 12 / (dn * np.sqrt(12)),
            "MaxDD_%": 100 * (wl / np.maximum.accumulate(wl) - 1).min()}


sr = lambda M: M.mean(-1) / M.std(-1, ddof=1) * np.sqrt(12)
VERS = ["Unmanaged", "Whole mix", "Legs, mkt unmanaged", "Legs, all managed"]
nets, rows = {}, []
for reg in REGIONS:
    D, M, fy, cal = load(reg)
    M = M.dropna().loc[:"2025-12"]; D = D.loc[D.dropna().index[0]:]
    leg = {}
    for f in FACS:   # each factor vol-managed on its own GARCH (US: same tags as panel_1998.py, so those caches are reused)
        if reg == "US":
            v = garch_monthly(f).reindex(M.index).dropna(); e = M[f].loc[v.index]
            w = pd.Series(exposure(v, solve_c(v.loc[cal[0]:cal[1]])), index=v.index)
            s = simulate(w, e, rf.reindex(e.index), 5e-4); s.index = e.index; leg[f] = s.net_ex
        else:
            leg[f] = managed(reg, f.replace("-", ""), M[f], D[f].fillna(0.0), fy, cal)
    for mix, fs in MIXES.items():
        tag = "_".join(f.replace("-", "") for f in fs)
        e = M[fs].mean(axis=1)
        wm = managed(reg, tag, e, D[fs].fillna(0.0).mean(axis=1), fy, cal)
        idx = wm.index.intersection(pd.concat([leg[f] for f in fs], axis=1).dropna().index)
        lm = pd.concat([M[f] if f == "Mkt-RF" else leg[f] for f in fs], axis=1).loc[idx].mean(axis=1)
        la = pd.concat([leg[f] for f in fs], axis=1).loc[idx].mean(axis=1)
        for ver, z in zip(VERS, (e.loc[idx], wm.loc[idx], lm, la)):
            nets[(reg, mix, ver)] = z
            for p, (a, b) in PERIODS.items():
                rows.append({"region": reg, "mix": mix, "version": ver, "period": p, **stats(z.loc[a:b], rf.loc[a:b])})
T = pd.DataFrame(rows); T.to_csv(OUT / "intl_ff_1998.csv", index=False)
pd.set_option("display.width", 250)


def boot_idx(n):
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); return ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]


for mix in MIXES:
    print(f"\n================ {mix}")
    for p, (a, b) in PERIODS.items():
        P = T[(T.mix == mix) & (T.period == p)].pivot(index="region", columns="version", values="Sharpe").loc[REGIONS, VERS]
        P.columns = ["SR unm", "SR whole", "SR legs mkt unm", "SR legs all"]
        DD = T[(T.mix == mix) & (T.period == p)].pivot(index="region", columns="version", values="MaxDD_%").loc[REGIONS, VERS]
        P["DD unm"], P["DD whole"], P["DD legs mkt unm"] = DD["Unmanaged"], DD["Whole mix"], DD["Legs, mkt unmanaged"]
        print(f"\n--- {p}  (net Sharpe; max drawdown %)")
        print(P.round(2).to_string())
        for ver in VERS[1:]:
            g = []
            for r in ("Developed_ex_US", "US"):
                m = nets[(r, mix, ver)].loc[a:b].to_numpy(); u = nets[(r, mix, "Unmanaged")].loc[a:b].to_numpy(); I = boot_idx(len(m))
                g.append((r, sr(m) - sr(u), *np.percentile(sr(m[I]) - sr(u[I]), [2.5, 97.5])))
            Mm = np.stack([nets[(r, mix, ver)].loc[a:b].to_numpy() for r in POOL])
            Mu = np.stack([nets[(r, mix, "Unmanaged")].loc[a:b].to_numpy() for r in POOL]); I = boot_idx(Mm.shape[1])
            gb = (sr(Mm[:, I]) - sr(Mu[:, I])).mean(0); g0 = sr(Mm) - sr(Mu); lo, hi = np.percentile(gb, [2.5, 97.5])
            txt = "  ".join(f"{r[:12]} {d:+.2f} [{l:+.2f},{h:+.2f}]" for r, d, l, h in g)
            print(f"  dSR {ver:20s} pooled 3: {g0.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] ({int((g0 > 0).sum())}/3) | {txt}")
