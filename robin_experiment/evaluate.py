"""Out-of-sample evaluation of the real-time backtest."""
import pickle, glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

KAPPA = 0.001          # 10 bp per unit of |dw|  (MM Table IV middle column)
GAMMA = 5.0
STRATS = ["RV", "garch_A", "garch_B", "gjr_A", "gjr_B", "lstm_A", "lstm_B"]
LABEL = {"Unmanaged": "Buy-and-hold", "RV": "MM: last month RV", "garch_A": "GARCH · A", "garch_B": "GARCH · B",
         "gjr_A": "GJR · A", "gjr_B": "GJR · B", "lstm_A": "GARCH-LSTM · A", "lstm_B": "GARCH-LSTM · B"}
PERIODS = {"Full OOS": (None, None), "1937-2015 (MM era)": (None, "2015-12"), "2016-2026 (post-publication)": ("2016-01", None)}


def load_all():
    out = []
    for p in sorted(glob.glob("results/bt_*.pkl")):
        out.append(pickle.load(open(p, "rb"))["months"])
    df = pd.concat(out)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


def strategy_returns(g):
    """g: one factor's monthly rows. Returns DataFrames of weights, gross and net returns."""
    g = g.set_index("month").sort_index()
    W, G, N = {}, {}, {}
    W["Unmanaged"] = pd.Series(1.0, index=g.index)
    G["Unmanaged"] = N["Unmanaged"] = g["f"]
    for s in STRATS:
        w = g[f"c_{s}"] / g[f"V_{s}"]
        W[s] = w
        G[s] = w * g["f"]
        N[s] = G[s] - KAPPA * w.diff().abs().fillna(0)
    return pd.DataFrame(W), pd.DataFrame(G), pd.DataFrame(N), g


def maxdd(x):
    cum = (1 + x).cumprod()
    return float((cum / cum.cummax() - 1).min())


def sharpe(x):
    return np.sqrt(12) * x.mean() / x.std()


def block_boot_dsr(a, b, n=2000, block=12, seed=0):
    """Two-sided p-value for SR(a) - SR(b) = 0 via a circular block bootstrap."""
    rng = np.random.default_rng(seed)
    T = len(a)
    X = np.column_stack([a, b])
    d0 = sharpe(pd.Series(a)) - sharpe(pd.Series(b))
    nb = int(np.ceil(T / block))
    ds = np.empty(n)
    for i in range(n):
        st = rng.integers(0, T, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        x = X[idx]
        ds[i] = np.sqrt(12) * (x[:, 0].mean() / x[:, 0].std(ddof=1) - x[:, 1].mean() / x[:, 1].std(ddof=1))
    se = ds.std(ddof=1)
    from scipy.stats import norm
    return d0, se, 2 * (1 - norm.cdf(abs(d0) / se))


def spanning(y, x):
    r = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC0")
    rmse = np.sqrt(r.mse_resid)
    return 1200 * r.params.iloc[0], r.tvalues.iloc[0], np.sqrt(12) * r.params.iloc[0] / rmse, r.params.iloc[1]


def realtime_combo(f, fs, min_obs=120):
    """Cederburg et al. (2020): each month, combine unmanaged and managed with mean-variance weights
    estimated on past OOS months only; returns the combination return series (from min_obs on)."""
    X = pd.concat([f, fs], axis=1).dropna().values
    out, idx = [], []
    ix = f.index
    for t in range(min_obs, len(X)):
        past = X[:t]
        mu, S = past.mean(0), np.cov(past.T)
        b = np.linalg.solve(S, mu)
        # scale so the combination has the unmanaged in-sample variance (Sharpe unaffected)
        b = b * np.sqrt(S[0, 0] / (b @ S @ b))
        out.append(X[t] @ b)
        idx.append(ix[t])
    return pd.Series(out, index=idx)


def table(df):
    rows = []
    for fac, g in df.groupby("factor", sort=False):
        W, G, N, g2 = strategy_returns(g)
        for pname, (a, b) in PERIODS.items():
            sl = slice(a, b)
            Gp, Np, Wp, gp = G.loc[sl], N.loc[sl], W.loc[sl], g2.loc[sl]
            if len(gp) < 24:
                continue
            for s in ["Unmanaged"] + STRATS:
                x, xn = Gp[s], Np[s]
                row = {"factor": fac, "period": pname, "strategy": s, "N": len(x),
                       "mean_%pa": 1200 * x.mean(), "vol_%pa": 100 * np.sqrt(12) * x.std(),
                       "SR": sharpe(x), "SR_net": sharpe(xn),
                       "CER_net_%pa": 1200 * (xn.mean() - GAMMA / 2 * xn.var()),
                       "maxDD_%": 100 * maxdd(x), "turnover": Wp[s].diff().abs().mean(),
                       "w_p50": Wp[s].median(), "w_p99": Wp[s].quantile(.99)}
                if s != "Unmanaged":
                    al, t, ar, be = spanning(x, Gp["Unmanaged"])
                    row.update({"alpha_%pa": al, "alpha_t": t, "appraisal": ar, "beta": be})
                    d, se, p = block_boot_dsr(x.values, Gp["Unmanaged"].values)
                    row.update({"dSR_vs_BH": d, "p_vs_BH": p})
                    V, rv = gp[f"V_{s}"], gp["rv"]
                    row.update({"QLIKE": float(np.mean(rv / V - np.log(rv / V) - 1)),
                                "corr_logV_logRV": float(np.corrcoef(np.log(V), np.log(rv))[0, 1])})
                rows.append(row)
    return pd.DataFrame(rows)


def ab_tests(df):
    """Objective B vs objective A, same model; each model vs the MM rule."""
    rows = []
    for fac, g in df.groupby("factor", sort=False):
        W, G, N, _ = strategy_returns(g)
        for pname, (a, b) in PERIODS.items():
            Gp = G.loc[slice(a, b)]
            if len(Gp) < 24:
                continue
            for m in ["garch", "gjr", "lstm"]:
                d, se, p = block_boot_dsr(Gp[f"{m}_B"].values, Gp[f"{m}_A"].values)
                rows.append({"factor": fac, "period": pname, "comparison": f"{m}: B - A", "dSR": d, "se": se, "p": p})
            for s in ["garch_A", "gjr_A", "lstm_A", "garch_B", "gjr_B", "lstm_B"]:
                d, se, p = block_boot_dsr(Gp[s].values, Gp["RV"].values)
                rows.append({"factor": fac, "period": pname, "comparison": f"{s} - RV", "dSR": d, "se": se, "p": p})
    return pd.DataFrame(rows)


def combo_table(df):
    rows = []
    for fac, g in df.groupby("factor", sort=False):
        W, G, N, _ = strategy_returns(g)
        for s in ["RV", "gjr_A", "gjr_B", "lstm_A", "lstm_B", "garch_A", "garch_B"]:
            c = realtime_combo(G["Unmanaged"], G[s])
            bh = G["Unmanaged"].loc[c.index]
            d, se, p = block_boot_dsr(c.values, bh.values)
            rows.append({"factor": fac, "strategy": s, "start": str(c.index[0]), "SR_combo": sharpe(c),
                         "SR_BH": sharpe(bh), "dSR": d, "p": p})
    return pd.DataFrame(rows)


def spike_table(df, q=0.95, horizon=6):
    """Spike test: a spike month is one whose realized variance exceeds the q-quantile of all RV
    observed up to then (real-time definition). For the `horizon` months after each spike we report the
    average weight relative to the strategy's own median weight (how far back in it is) and the
    Sharpe ratio / mean of the strategy over those months (was re-entering right?)."""
    rows = []
    for fac, g in df.groupby("factor", sort=False):
        W, G, N, g2 = strategy_returns(g)
        rv = g2["rv"]
        thr = rv.expanding(60).quantile(q).shift(1)
        spikes = rv.index[(rv > thr).values]
        post = set()
        for s in spikes:
            for h in range(1, horizon + 1):
                post.add(s + h)
        post = sorted(p for p in post if p in rv.index and p not in set(spikes))
        for s in ["Unmanaged"] + STRATS:
            x = G.loc[post, s]
            rows.append({"factor": fac, "strategy": s, "n_spikes": len(spikes), "n_post": len(post),
                         "rel_weight_post": float((W.loc[post, s] / W[s].median()).mean()),
                         "mean_post_%pa": 1200 * x.mean(), "SR_post": sharpe(x)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    df = load_all()
    t = table(df); t.to_csv("results/oos_table.csv", index=False)
    ab = ab_tests(df); ab.to_csv("results/ab_tests.csv", index=False)
    cb = combo_table(df); cb.to_csv("results/combo_table.csv", index=False)
    cols = ["factor", "strategy", "N", "SR", "SR_net", "alpha_%pa", "alpha_t", "dSR_vs_BH", "p_vs_BH", "QLIKE", "corr_logV_logRV", "turnover", "w_p99"]
    for p in PERIODS:
        print("\n=====", p); print(t[t.period == p][cols].round(3).to_string(index=False))
    print("\n===== B vs A and model vs RV (Sharpe differences, block bootstrap)")
    print(ab.round(3).to_string(index=False))
    print("\n===== Real-time combination (Cederburg)"); print(cb.round(3).to_string(index=False))

    sp = spike_table(df); sp.to_csv("results/spike_table.csv", index=False)
    print("\n===== Post-spike behaviour"); print(sp.round(3).to_string(index=False))
