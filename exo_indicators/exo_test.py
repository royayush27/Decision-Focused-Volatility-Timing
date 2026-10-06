"""Do exogenous indicators (credit spreads, news attention, news sentiment) warn of volatility early?

Market (Mkt-RF). Row t = month being forecast; forecast made at the end of month t-1.
Indicator timing (strict, no look-ahead):
  credit / term spreads, EPU : value for month t-1 (monthly average of daily data, known at end of t-1)
  EMV, news sentiment        : value for month t-2 (published with a lag of days to a week)
Target: log realised variance of month t. Baseline: OLS of log RV_t on log GARCH_t and log RV_{t-1}.
Augmented: baseline + one indicator (or all). Expanding window, refit every month.
"""
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

S = "exo_indicators/"
fc = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")
rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"].reindex(fc.index)
ind = pd.read_csv(S + "indicators_monthly.csv", index_col="month")
_v = pd.read_csv("data/VIXCLS.csv")
_v.columns = ["date", "vix"]; _v["date"] = pd.to_datetime(_v["date"]); _v = _v.dropna()
vix = (_v.set_index("date")["vix"].resample("ME").last() ** 2 / 1e4 / 12)          # end-of-month VIX^2, monthly variance
vix.index = vix.index.strftime("%Y-%m")

L1 = ind.shift(1).reindex(fc.index)   # value of month t-1, placed on row t
L2 = ind.shift(2).reindex(fc.index)   # value of month t-2
X = pd.DataFrame(index=fc.index)
X["DEF"] = L1["DEF"]                                   # Baa - Aaa
X["dDEF"] = L1["DEF"] - L2["DEF"]                      # one-month change in Baa - Aaa
X["CS"] = L1["CS"]                                     # Baa - 10y Treasury
X["dCS"] = L1["CS"] - L2["CS"]
X["TERM"] = L1["TERM"]                                 # 10y - 3m
X["lEPU"] = np.log(L1["EPU"])                          # news-based policy uncertainty (daily avg)
X["lEMV"] = np.log(L2["EMV"])                          # news-based equity-volatility attention
X["NS"] = L2["NS_avg"]                                 # SF Fed news sentiment (tone)
X["dNS"] = L2["NS_avg"] - ind["NS_avg"].shift(3).reindex(fc.index)
X["lVIX2"] = np.log(vix.shift(1).reindex(fc.index))    # benchmark: option-implied, end of t-1

y = np.log(fc["rv"])
B = pd.DataFrame({"lg": np.log(fc["garch_rv_forecast"]), "lr": np.log(fc["rv_lag1"])}, index=fc.index)
ex, rv = fc["ret_m"], fc["rv"]


def oos(cols, est_start, oos_start, oos_end="2025-12"):
    """Expanding-window OLS forecasts of log RV with baseline regressors + cols."""
    Z = pd.concat([B, X[cols]], axis=1) if cols else B.copy()
    Z.insert(0, "c", 1.0)
    out = pd.Series(np.nan, index=fc.index); s2 = pd.Series(np.nan, index=fc.index)
    months = fc.loc[oos_start:oos_end].index
    for t in months:
        tr = Z.loc[est_start:t].iloc[:-1]
        yt = y.loc[tr.index]
        ok = tr.notna().all(axis=1) & yt.notna()
        A, b = tr[ok].to_numpy(), yt[ok].to_numpy()
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = b - A @ beta
        z = Z.loc[t]
        if z.notna().all() and ok.sum() >= 48:
            out[t] = z.to_numpy() @ beta
            s2[t] = res.var(ddof=A.shape[1])
    return out, s2


def clark_west(e0, e1, f0, f1, lags=3):
    f = e0 ** 2 - e1 ** 2 + (f0 - f1) ** 2
    f = f - 0; n = len(f); m = f.mean(); u = f - m
    g = [np.mean(u[k:] * u[:n - k]) for k in range(lags + 1)]
    lrv = g[0] + 2 * sum((1 - k / (lags + 1)) * g[k] for k in range(1, lags + 1))
    return m / np.sqrt(lrv / n)


def qlike(a, f):
    q = a / f
    return (q - np.log(q) - 1).mean()


def evaluate(est_start, oos_start, ev, cal, cands, label):
    print(f"\n========== {label}: estimation from {est_start}, OOS forecasts from {oos_start}, "
          f"c calibrated {cal[0]}..{cal[1]}, evaluation {ev[0]}..{ev[1]}")
    base, s2b = oos([], est_start, oos_start)
    fcs = {"baseline": (base, s2b)}
    for c in cands:
        fcs[c] = oos([c], est_start, oos_start)
    fcs["all_exo"] = oos([c for c in cands if c != "lVIX2"], est_start, oos_start)
    # equal-weight combination of single-indicator forecasts (Paye 2012: combinations do best)
    singles = [c for c in cands if c != "lVIX2"]
    fcs["combo"] = (pd.concat([fcs[c][0] for c in singles], axis=1).mean(axis=1),
                    pd.concat([fcs[c][1] for c in singles], axis=1).mean(axis=1))
    a, b = ev
    yy = y.loc[a:b]
    surprise = (y - B["lg"]).loc[a:b]                      # GARCH surprise, log
    spikes = surprise >= surprise.quantile(0.95)
    big = yy >= yy.quantile(0.90)
    rows, nets = [], {}
    rng = np.random.default_rng(0)
    for k, (f, s2) in fcs.items():
        v = np.exp(f + s2 / 2)                             # variance forecast
        e1 = (yy - f.loc[a:b]).to_numpy(); e0 = (yy - base.loc[a:b]).to_numpy()
        rev = (f - base).loc[a:b]
        w = pd.Series(exposure(v, solve_c(v.loc[cal[0]:cal[1]].dropna())), index=fc.index)
        w = w.fillna(1.0)
        sim = simulate(w, ex, rf, 5e-4); sim.index = fc.index
        W = w.loc[a:b]; x = W / W.mean(); R = ex.loc[a:b]; Q = rv.loc[a:b]
        z = sim.net_ex.loc[a:b]; nets[k] = z.to_numpy()
        rows.append({"model": k,
                     "R2oos_vs_base": 1 - (e1 ** 2).sum() / (e0 ** 2).sum(),
                     "CW_t": np.nan if k == "baseline" else clark_west(e0, e1, base.loc[a:b].to_numpy(), f.loc[a:b].to_numpy()),
                     "QLIKE": qlike(Q, v.loc[a:b]),
                     "rev_in_spikes": rev[spikes].mean(), "rev_other": rev[~spikes].mean(),
                     "rev_before_top10%": rev[big].mean(),
                     "VR": (Q * x ** 2).mean() / Q.mean(), "MT": (x * R).mean() / R.mean(),
                     "SR_net": z.mean() / z.std() * np.sqrt(12)})
    T = pd.DataFrame(rows).set_index("model")
    print(f"months: {len(yy)}, GARCH-surprise spikes: {int(spikes.sum())}  (rev = forecast revision vs baseline, log points)")
    print(T.round(3).to_string())
    # paired bootstrap of net Sharpe difference vs baseline
    n = len(nets["baseline"]); BL, BB = 6, 4000
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL))))
    I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    sr = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
    zb = nets["baseline"]; out = []
    for k in nets:
        if k == "baseline":
            continue
        d = sr(nets[k][I]) - sr(zb[I]); d0 = (nets[k].mean() / nets[k].std(ddof=1) - zb.mean() / zb.std(ddof=1)) * np.sqrt(12)
        out.append(f"{k}: dSR {d0:+.3f} [{np.percentile(d - d.mean() + d0, 2.5):+.3f}, {np.percentile(d - d.mean() + d0, 97.5):+.3f}]")
    print("Net Sharpe vs baseline (95% block-bootstrap CI):\n  " + "\n  ".join(out))
    return T, fcs


if __name__ == "__main__":
    # Panel A: long history, spreads only
    evaluate("1970-01", "1985-01", ("2000-01", "2024-12"), ("1985-01", "1999-12"),
             ["DEF", "dDEF", "CS", "dCS", "TERM"], "Panel A (spreads, long history)")
    # Panel B: news era, all indicators plus VIX as the option-implied benchmark
    evaluate("1986-03", "1995-01", ("2005-01", "2024-12"), ("1995-01", "2004-12"),
             ["DEF", "dDEF", "CS", "dCS", "lEPU", "lEMV", "NS", "dNS", "lVIX2"], "Panel B (news era)")
