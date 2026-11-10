"""Robustness checks behind the Responses to Questions (numbers quoted there), run on the final notebook's data,
forecasts and backtests. Run from this folder:  python qa_checks.py   (a few minutes; bootstrap with 5,000 draws)."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr
from nb_loader import load_notebook

g = load_notebook()
globals().update({k: v for k, v in g.items() if not k.startswith("__")})
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
A, B = PERIODS[MAIN]; u = nets["Buy and hold"]; K4 = ["RV", "GARCH", "GJR", "LSTM"]
r3 = lambda d: {k: (round(float(v), 3) if np.isscalar(v) else v) for k, v in d.items()}
bt = lambda w: backtest(w.dropna().loc["1970-01":], mkt)


def stats(z, a=A, b=B):
    z = z.loc[a:b]
    return {"Sharpe": sharpe(z), "vol %": 100 * z.std() * 12 ** .5, "max DD %": 100 * max_drawdown(z, rf.loc[a:b]), "vol of vol %": vol_of_vol(z)}


print("=== Q1: simple past-only stock/cash rules, 1998-2025")
rv_hist = forecasts["RV"]
rules = {"Cash if RV in top 20%": (rv_hist <= rv_hist.expanding(120).quantile(0.8)).astype(float).where(rv_hist.expanding(120).quantile(0.8).notna()),
         "Cash if RV in top 30%": (rv_hist <= rv_hist.expanding(120).quantile(0.7)).astype(float).where(rv_hist.expanding(120).quantile(0.7).notna()),
         "Cash after negative 12-month return": (mkt.rolling(12).sum().shift(1) > 0).astype(float).where(mkt.rolling(12).sum().shift(1).notna())}
for name, w in rules.items():
    z = bt(w)
    print(name, r3(stats(z)), "p vs B&H", round(bootstrap_test(z.loc[A:B], u.loc[A:B])[3], 2),
          "p vs timed RV", round(bootstrap_test(z.loc[A:B], nets[TIMED["RV"]].loc[A:B])[3], 2))

print("\n=== Q8: fixed mix with the same average exposure")
for k in K4:
    t = TIMED[k]; z = nets[t].loc[A:B]; idx = W[t].loc["1970-01":].dropna().index
    avg = W[t].loc[A:B].mean(); lev = z.std() / u.loc[A:B].std()
    fx = backtest(pd.Series(avg, index=idx), mkt).loc[A:B]; fx2 = backtest(pd.Series(lev, index=idx), mkt).loc[A:B]
    print(k, f"avg exposure {avg:.2f}", "fixed", r3(stats(fx)), f"| scaled to timed vol ({lev:.2f})", r3(stats(fx2)),
          "| timed", r3(stats(z)), "| p Sharpe timed vs fixed", round(bootstrap_test(z, fx)[3], 2))


def c_match(v, f, ratio, cap):
    v, f = np.asarray(v, float), np.asarray(f, float); target = ratio * f.std(ddof=1)
    gap = lambda c: (np.clip(c / v, 0, cap) * f).std(ddof=1) - target
    hi = target / (f / v).std(ddof=1)
    while gap(hi) < 0:
        hi *= 2
    return brentq(gap, 1e-12, hi, xtol=1e-16)


def weights_ratio(v, f, ratio, cap):
    v = v.astype(float); f = f.reindex(v.index); ok = v.notna() & f.notna(); c = pd.Series(np.nan, index=v.index)
    for y in sorted({m[:4] for m in v.index[v.notna()]}):
        past = ok & (v.index < f"{y}-01")
        if past.sum() >= MIN_MONTHS:
            c[v.index.str[:4] == y] = c_match(v[past], f[past], ratio, cap)
    return (c / v).clip(0, cap)


print("\n=== Q5: no leverage")
for k in K4:
    w1 = W[TIMED[k]].clip(upper=1.0); z1 = bt(w1).loc[A:B]
    z2 = bt(weights_ratio(forecasts[k], mkt, 0.69, 1.0)).loc[A:B]
    w3 = timing_weights(forecasts[k], mkt, cap=1.0)
    print(k, f"(i) cut at 1: avg exposure {w1.loc[A:B].mean():.2f}", r3(stats(z1)), "gain/p", r3(dict(zip(["gain", "lo", "hi", "p"], bootstrap_test(z1, u.loc[A:B])))),
          "| (ii) target 0.69:", r3(stats(z2)), f"| cap 1 with vol-matched c: avg exposure {w3.loc[A:B].mean():.3f}, Sharpe {sharpe(bt(w3).loc[A:B]):.3f}")

print("\n=== Q11: caps (c re-matched each January)")
for cap in [1.25, 1.5, 2.0, None]:
    for k in K4:
        z = bt(timing_weights(forecasts[k], mkt, cap=cap)).loc[A:B]
        print(cap, k, r3(stats(z)), "gain/p", r3(dict(zip(["gain", "lo", "hi", "p"], bootstrap_test(z, u.loc[A:B])))))

print("\n=== Q3: 1975-1997 and 1975-2025")
for a, b in [("1975-01", "1997-12"), ("1975-01", "2025-12")]:
    uu = u.loc[a:b]; print(a, b, "buy and hold Sharpe", round(sharpe(uu), 3))
    for k in K4:
        z = nets[TIMED[k]].loc[a:b]
        out = {"Sharpe": sharpe(z)}
        out["Sharpe gain"], _, _, out["p"] = bootstrap_test(z, uu)
        e, _, _, out["DD p"] = bootstrap(paired(z, uu, rf.loc[a:b]), drawdown_gain_equal_vol); out["DD gain pts"] = 100 * e
        out["vol of vol gain"], _, _, out["vov p"] = bootstrap(paired(z, uu), volvol_gain)
        print(" ", k, r3(out))

print("\n=== Q10: Sharpe gain by start year (to 2025) and vol-of-vol p by window")
for y in [1975, 1985, 1995, 1998, 2000, 2010]:
    a = f"{y}-01"; print(y, {k: round(sharpe(nets[TIMED[k]].loc[a:B]) - sharpe(u.loc[a:B]), 3) for k in K4})
for a, b in [("1998-01", "2025-12"), ("1985-01", "2025-12"), ("1975-01", "2025-12"), ("2016-01", "2025-12")]:
    print("vol of vol (gain, p)", a, b, {k: tuple(round(float(x), 3) for x in np.array(bootstrap(paired(nets[TIMED[k]].loc[a:b], u.loc[a:b]), volvol_gain))[[0, 3]]) for k in ["GARCH", "GJR", "LSTM"]})

print("\n=== Q4: 2000-02 to 2020-12")
a, b = "2000-02", "2020-12"
for k in FORECASTS:
    z = nets[TIMED[k]].loc[a:b]
    print(k, f"Sharpe {sharpe(z):.3f} vs {sharpe(u.loc[a:b]):.3f}", f"bootstrap p {bootstrap_test(z, u.loc[a:b])[3]:.2f}", f"Ledoit-Wolf p {ledoit_wolf_test(z, u.loc[a:b])[2]:.2f}")

print("\n=== Q19: drawdown gain at equal volatility, 2016-2025")
a, b = PERIODS["2016-2025"]
for k in K4:
    e, _, _, p = bootstrap(paired(nets[TIMED[k]].loc[a:b], u.loc[a:b], rf.loc[a:b]), drawdown_gain_equal_vol)
    print(k, round(100 * e, 1), "p", round(p, 2))

print("\n=== Q6: recovery months in 1998-2025 and the recovery fix")
rec = (state == "Recovery").loc[A:B]
print("recovery months", int(rec.sum()), "mean %", round(100 * mkt.loc[A:B][rec].mean(), 2), "sd %", round(100 * mkt.loc[A:B][rec].std(), 2))
zf = nets["Recovery fix"].loc[A:B]
print("fix:", r3(stats(zf)), "mean excess %", round(1200 * zf.mean(), 2), "| plain RV:", r3(stats(nets[TIMED['RV']])), "mean excess %", round(1200 * nets[TIMED['RV']].loc[A:B].mean(), 2),
      "| p", round(bootstrap_test(zf, nets[TIMED["RV"]].loc[A:B])[3], 2))

print("\n=== Q13 and Q16: accuracy ranking vs Sharpe ranking (Spearman; QLIKE lower = more accurate)")
sh = pd.Series({k: sharpe(nets[TIMED[k]].loc[A:B]) for k in FORECASTS})
print("Sharpe", sh.round(4).to_dict(), "| QLIKE", accuracy.round(3).to_dict())
print("rank correlation, accuracy vs Sharpe: 5 forecasts", round(-spearmanr(accuracy[FORECASTS], sh[FORECASTS])[0], 2),
      "3 models", round(-spearmanr(accuracy[MODELS], sh[MODELS])[0], 2))
WW = pd.DataFrame({k: W[TIMED[k]].loc[A:B] for k in FORECASTS}); capped = WW >= CAP - 1e-9; none = ~capped.any(axis=1)
rvA = rv_eval["Mkt-RF"].loc[A:B]
q_unc = pd.Series({k: qlike(forecasts[k].loc[A:B][none], rvA[none]) for k in FORECASTS})
perr = pd.Series({k: float(((WW[k] - (scales[k].loc[A:B] / rvA).clip(0, CAP)) ** 2).mean()) for k in FORECASTS})
print("months with no forecast at the cap:", int(none.sum()), "| months at the cap:", capped.sum().to_dict())
print("QLIKE in those months", q_unc.round(3).to_dict(), "rank corr", round(-spearmanr(q_unc, sh[FORECASTS])[0], 2))
print("position error", perr.round(3).to_dict(), "rank corr", round(-spearmanr(perr, sh[FORECASTS])[0], 2))

print("\n=== Q7: seed stability and sub-period signs of Dec minus Acc (GARCH-LSTM)")
for f, gg in lstm_files.items():
    x = gg.loc[A:B]
    sd = {o: np.log(x[[f"V_zhao_{o}_s{i}" for i in range(N_SEEDS)]]).std(axis=1).mean() for o in "AB"}
    ss = {o: np.std([sharpe(backtest(gg[f"W_zhao_{o}_s{i}"], gg["f"]).loc[A:B]) for i in range(N_SEEDS)], ddof=1) for o in "AB"}
    halves = [sharpe(dfl_nets[(f, "LSTM Dec")].loc[a:b]) - sharpe(dfl_nets[(f, "LSTM Acc")].loc[a:b]) for a, b in (PERIODS["1998-2015"], PERIODS["2016-2025"])]
    print(f, "cross-seed sd of log forecast Acc/Dec", round(sd["A"], 3), round(sd["B"], 3), "| sd of per-seed Sharpe Acc/Dec", round(ss["A"], 3), round(ss["B"], 3),
          "| Dec - Acc Sharpe 1998-2015, 2016-2025", [round(h, 3) for h in halves])

print("\n=== Q12: separate timing vs one rule on the factor mix, 1998-2025")
tf = pd.DataFrame({s: backtest(timing_weights(garch_monthly(s, daily_ret[s]).reindex(ret.index), ret[s]).loc["1970-01":], ret[s]) for s in FACTORS}).loc[A:B]
dmix = daily_ret[FACTORS].dropna().mean(axis=1)
rv_mix = (dmix ** 2).groupby(dmix.index.to_period("M")).sum(); rv_mix.index = rv_mix.index.astype(str)
mix_timed = backtest(timing_weights(rv_mix.shift(1).reindex(ret.index), ew_factors).loc["1970-01":], ew_factors)
vol36 = ret[FACTORS].rolling(36).std().shift(1); w_rp = (1 / vol36).div((1 / vol36).sum(axis=1), axis=0)
rp = (w_rp * ret[FACTORS]).sum(axis=1).where(w_rp.notna().all(axis=1)).dropna()
for name, z in {"EW factors, untimed": ew_factors, "average of 6 separately timed": tf.mean(axis=1), "one rule on the mix": mix_timed, "inverse-vol risk parity": rp}.items():
    print(name, r3(stats(z)))
print("p, separately timed vs untimed:", round(bootstrap_test(tf.mean(axis=1), ew_factors.loc[A:B])[3], 2),
      "| average pairwise correlation:", round(ret[FACTORS].loc[A:B].corr().values[np.triu_indices(6, 1)].mean(), 3))

if crsp_ex is not None:
    print("\n=== Q9: equal-weighted CRSP benchmark, sensitivity (2000-02 to 2020-12)")
    sr_ = lambda x: round(sharpe((x - rf).dropna().loc["2000-02":"2020-12"]), 3)
    print("baseline", sr_(stock_ret.where(held).mean(axis=1)), "| price 2 months back", sr_(stock_ret.where(held & prices.shift(2).notna()).mean(axis=1)),
          "| winsorise 1/99", sr_(stock_ret.where(held).clip(lower=stock_ret.where(held).quantile(0.01, axis=1), upper=stock_ret.where(held).quantile(0.99, axis=1), axis=0).mean(axis=1)),
          "| median stock", sr_(stock_ret.where(held).median(axis=1)))
