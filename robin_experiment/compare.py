"""Point 2: direct, pre-specified comparisons with multiple-testing correction.

Strategies (all real time, 1937/1974-2026, net of 10 bp per unit of |dw|):
  BH, RV (Moreira-Muir), GARCH A/B, GJR A/B,
  OVL A/B   = our first GARCH-LSTM (GJR x exp(LSTM) overlay), 2-seed ensemble
  ZHAO A/B  = Zhao et al. (2024) GARCH-LSTM, 5-seed ensemble
An ensemble averages the seeds' portfolio weights (each seed has its own real-time c).

Family of tests (Sharpe difference, net of costs), for each of the 6 factors:
  Q1 neural part      : ZHAO A - GJR A
  Q2 decision training: ZHAO B - ZHAO A,  GJR B - GJR A,  GARCH B - GARCH A
  Q3 full package     : ZHAO B - GJR A,  ZHAO B - RV
= 36 tests. p-values from a circular block bootstrap (12-month blocks, 5,000 draws, same draws for every
test so the dependence between tests is kept), adjusted by Holm (family-wise error), Benjamini-Hochberg
(false discovery rate) and the Romano-Wolf step-down max-t (family-wise, uses the joint bootstrap).
Forecast accuracy: Diebold-Mariano test on monthly QLIKE losses against realised variance.
"""
import glob, pickle
import numpy as np
import pandas as pd
from scipy.stats import norm

KAPPA = 0.001
FACTORS = ["Mkt", "SMB", "HML", "Mom", "RMW", "CMA"]
TESTS = [("Q1 neural part", "ZHAO_A", "GJR_A"),
         ("Q2 decision", "ZHAO_B", "ZHAO_A"), ("Q2 decision", "GJR_B", "GJR_A"), ("Q2 decision", "GARCH_B", "GARCH_A"),
         ("Q3 full package", "ZHAO_B", "GJR_A"), ("Q3 full package", "ZHAO_B", "RV")]


def weights_from(g, keys):
    """Average of the seeds' weights c/V."""
    return sum(g[f"c_{k}"] / g[f"V_{k}"] for k in keys) / len(keys)


def ensemble_V(g, keys):
    return sum(g[f"V_{k}"] for k in keys) / len(keys)


def build(fac):
    base = pickle.load(open(f"results/bt_{fac}.pkl", "rb"))["months"].set_index("month").sort_index()
    base.index = pd.PeriodIndex(base.index, freq="M")
    g = base.copy()
    try:
        s1 = pickle.load(open(f"results/seeds/bt_{fac}_seed1000.pkl", "rb"))["months"].set_index("month").sort_index()
        s1.index = pd.PeriodIndex(s1.index, freq="M")
        for obj in "AB":
            g[f"V_lstm1_{obj}"] = s1[f"V_lstm_{obj}"]
            g[f"c_lstm1_{obj}"] = s1[f"c_lstm_{obj}"]
        ovl = {o: [f"lstm_{o}", f"lstm1_{o}"] for o in "AB"}
    except FileNotFoundError:
        ovl = {o: [f"lstm_{o}"] for o in "AB"}
    try:
        z = pickle.load(open(f"results/zhao/bt_{fac}.pkl", "rb"))["months"].set_index("month").sort_index()
        z.index = pd.PeriodIndex(z.index, freq="M")
        zcols = [c for c in z.columns if c.startswith(("V_zhao", "c_zhao"))]
        g = g.join(z[zcols])
        zk = {o: sorted({c[2:] for c in zcols if c.startswith(f"V_zhao_{o}_")}) for o in "AB"}
    except FileNotFoundError:
        zk = None
    W, V = {"BH": pd.Series(1.0, index=g.index)}, {}
    for name, key in [("RV", "RV"), ("GARCH_A", "garch_A"), ("GARCH_B", "garch_B"), ("GJR_A", "gjr_A"), ("GJR_B", "gjr_B")]:
        W[name] = g[f"c_{key}"] / g[f"V_{key}"]
        V[name] = g[f"V_{key}"]
    for o in "AB":
        W[f"OVL_{o}"] = weights_from(g, ovl[o]); V[f"OVL_{o}"] = ensemble_V(g, ovl[o])
        if zk:
            W[f"ZHAO_{o}"] = weights_from(g, zk[o]); V[f"ZHAO_{o}"] = ensemble_V(g, zk[o])
            for k in zk[o]:   # single seeds, for the seed-spread table
                W[k] = g[f"c_{k}"] / g[f"V_{k}"]
    W = pd.DataFrame(W)
    gross = W.mul(g["f"], axis=0)
    net = gross - KAPPA * W.diff().abs().fillna(0)
    return net, gross, W, pd.DataFrame(V), g["rv"], zk


def sr(x):
    return np.sqrt(12) * x.mean(0) / x.std(0, ddof=1)


def boot_idx(T, n, block, rng):
    nb = int(np.ceil(T / block))
    st = rng.integers(0, T, (n, nb))
    return ((st[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]) % T


def run_family(nets, n_boot=5000, block=12, seed=0):
    """nets: {factor: net-return DataFrame}. Returns one row per test with raw and adjusted p-values."""
    rng = np.random.default_rng(seed)
    rows, dstar = [], []
    for fac in FACTORS:
        X = nets[fac]
        T = len(X)
        idx = boot_idx(T, n_boot, block, rng)
        sr0 = sr(X)
        # bootstrap Sharpe ratios of every strategy on the same draws
        Xv = X.values
        B = np.empty((n_boot, X.shape[1]))
        for b in range(n_boot):
            x = Xv[idx[b]]
            B[b] = np.sqrt(12) * x.mean(0) / x.std(0, ddof=1)
        B = pd.DataFrame(B, columns=X.columns)
        for q, a, c in TESTS:
            d = sr0[a] - sr0[c]
            db = (B[a] - B[c]).values
            se = db.std(ddof=1)
            rows.append({"question": q, "factor": fac, "test": f"{a} - {c}", "SR_a": sr0[a], "SR_b": sr0[c],
                         "dSR": d, "se": se, "t": d / se, "p_raw": 2 * (1 - norm.cdf(abs(d / se)))})
            dstar.append((db - d) / se)            # centred, studentised bootstrap draws
    res = pd.DataFrame(rows)
    # Holm
    m = len(res)
    order = np.argsort(res.p_raw.values)
    holm = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, min(1, (m - rank) * res.p_raw.values[i]))
        holm[i] = run
    res["p_holm"] = holm
    # Benjamini-Hochberg
    bh = np.empty(m); run = 1
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        run = min(run, res.p_raw.values[i] * m / (rank + 1))
        bh[i] = run
    res["p_bh"] = bh
    # Romano-Wolf step-down (two-sided, max |t|); draws are independent across factors (different
    # samples), so the joint distribution is built by pairing draw b of every factor
    Tstar = np.abs(np.vstack(dstar))              # (m, n_boot)
    tobs = np.abs(res.t.values)
    ordr = np.argsort(-tobs)
    rw = np.empty(m); run = 0
    for k, i in enumerate(ordr):
        rest = ordr[k:]
        p = (Tstar[rest].max(0) >= tobs[i]).mean()
        run = max(run, p)
        rw[i] = run
    res["p_romano_wolf"] = rw
    return res


def dm_test(l1, l2, lags=6):
    """Diebold-Mariano on loss differential l1 - l2 (negative mean = model 1 more accurate), Newey-West."""
    d = (l1 - l2).dropna().values
    T = len(d); dm = d.mean(); e = d - dm
    s = e @ e / T
    for k in range(1, lags + 1):
        s += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / T
    t = dm / np.sqrt(s / T)
    return dm, t, 2 * (1 - norm.cdf(abs(t)))


def qlike(rv, V):
    x = rv / V
    return x - np.log(x) - 1


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    nets, grosses, accs, spreads, srt = {}, {}, [], [], []
    for fac in FACTORS:
        net, gross, W, V, rv, zk = build(fac)
        nets[fac] = net
        main = ["BH", "RV", "GARCH_A", "GARCH_B", "GJR_A", "GJR_B", "OVL_A", "OVL_B", "ZHAO_A", "ZHAO_B"]
        srt.append({"factor": fac, **{f"{k}": sr(net[k]) for k in main if k in net},
                    **{f"gross_{k}": sr(gross[k]) for k in main if k in gross}})
        for k in [c for c in V.columns]:
            accs.append({"factor": fac, "model": k, "QLIKE": qlike(rv, V[k]).mean(),
                         "corr_log": np.corrcoef(np.log(V[k]), np.log(rv))[0, 1]})
        for a, b in [("ZHAO_A", "GJR_A"), ("ZHAO_B", "ZHAO_A"), ("OVL_A", "GJR_A")]:
            if a in V and b in V:
                dm, t, p = dm_test(qlike(rv, V[a]), qlike(rv, V[b]))
                accs.append({"factor": fac, "model": f"DM {a} vs {b}", "QLIKE": dm, "corr_log": np.nan, "DM_t": t, "DM_p": p})
        if zk:
            for o in "AB":
                s = [sr(net[k]) for k in zk[o]]
                spreads.append({"factor": fac, "objective": o, "seed_min": min(s), "seed_max": max(s),
                                "seed_sd": np.std(s, ddof=1), "ensemble": sr(net[f"ZHAO_{o}"])})
    srt = pd.DataFrame(srt); srt.to_csv("results/compare_sharpe.csv", index=False)
    print("Net Sharpe ratios\n", srt[[c for c in srt.columns if not c.startswith("gross")]].round(3).to_string(index=False))
    acc = pd.DataFrame(accs); acc.to_csv("results/compare_accuracy.csv", index=False)
    print("\nForecast accuracy (monthly QLIKE vs realised variance; DM rows: mean loss difference)\n", acc.round(4).to_string(index=False))
    if spreads:
        sp = pd.DataFrame(spreads); sp.to_csv("results/compare_seed_spread.csv", index=False)
        print("\nSeed spread of the Zhao GARCH-LSTM (net Sharpe)\n", sp.round(3).to_string(index=False))
    if all("ZHAO_A" in nets[f] for f in FACTORS):
        fam = run_family({f: nets[f] for f in FACTORS})
        fam.to_csv("results/compare_family.csv", index=False)
        print("\nFamily of 36 tests\n", fam.round(3).to_string(index=False))
