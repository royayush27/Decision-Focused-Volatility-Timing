# archived in old/scipy_garch: run from anywhere, paths below are relative to the project root
import os as _os, sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
for _p in (_ROOT, _P(__file__).resolve().parent):
    _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)

import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BASE_KAPPA, GAMMA, PERIODS, metrics, simulate
from calibrate_c import CAL_END, exposure, solve_c
from compare_dist import dm, qlike

RES = Path("results_1970")
FC = Path("results_1970/forecasts")
MAIN = re.compile(r"^(garch|gjr|ginn)_(stat|dec)$")
ACC_PERIODS = {"cal 1970-99": ("1970-01", "1999-12"), "dev 2000-24": ("2000-01", "2024-12"),
               "2025": ("2025-01", "2025-12")}
TEST_PERIOD = "dev 2000-2024"
BLOCK, REPS, SEED = 6, 5000, 0


def load():
    base = pd.read_csv(RES / "market_forecasts.csv", index_col="month")
    rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"].reindex(base.index)
    v = {"Lagged RV": base["rv_baseline_forecast"]}
    for name in sorted(p.stem for p in FC.glob("*.csv") if MAIN.match(p.stem)):
        v[name] = pd.read_csv(FC / f"{name}.csv", index_col="month")["v_hat"]
    for k, s in v.items():
        assert s.index.equals(base.index) and s.notna().all() and (s > 0).all(), k
    assert rf.notna().all()
    return base, rf, v


def accuracy(base, v):
    rows = []
    for label, (a, b) in ACC_PERIODS.items():
        rv = base.rv.loc[a:b]
        for k, s in v.items():
            f = s.loc[a:b]
            row = {"period": label, "model": k, "QLIKE": qlike(rv, f).mean(),
                   "MSE_e5": ((rv - f) ** 2).mean() * 1e5, "bias": f.mean() / rv.mean()}
            if k != "Lagged RV" and len(rv) >= 60:
                row["DM_t_vs_RV"], row["DM_p"] = dm(qlike(rv, f) - qlike(rv, v["Lagged RV"].loc[a:b]))
            rows.append(row)
    return pd.DataFrame(rows).set_index(["period", "model"])


def make_weights(v):
    c = {k: solve_c(s.loc[:CAL_END]) for k, s in v.items()}
    index = v["Lagged RV"].index
    w = pd.DataFrame({k: exposure(s, c[k]) for k, s in v.items()}, index=index)
    w.insert(0, "Unmanaged", 1.0)
    return c, w


def backtest(base, rf, w, kappa=BASE_KAPPA):
    ex, sims, rows = base.ret_m, {}, []
    for k in w:
        sim = simulate(w[k], ex, rf, kappa)
        sim.index = w.index
        sims[k] = sim
    for label, (a, b) in PERIODS.items():
        for k, sim in sims.items():
            rows.append(metrics(sim.loc[a:b]).rename((label, k)))
        for k in w.columns[1:]:
            avg = sims[k].loc[a:b, "w"].mean()
            idx = ex.loc[a:b].index
            sim = simulate(pd.Series(avg, index=idx), ex.loc[a:b], rf.loc[a:b], kappa)
            rows.append(metrics(sim).rename((label, f"Static {avg:.2f} ({k})")))
    table = pd.DataFrame(rows)
    table.index.names = ["period", "strategy"]
    return table, sims


def block_idx(n, block, rng):
    starts = rng.integers(0, n, int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)) % n).ravel()[:n]


def util_sharpe(x, gamma=GAMMA):
    mu, var = x.mean() * 12, x.var(ddof=1) * 12
    return mu - gamma / 2 * var, mu / np.sqrt(var)


def boot_diff(a, b, block=BLOCK, reps=REPS, seed=SEED):
    n, rng = len(a), np.random.default_rng(seed)
    ua, sa = util_sharpe(a)
    ub, sb = util_sharpe(b)
    d_u, d_s = ua - ub, sa - sb
    bu, bs = np.empty(reps), np.empty(reps)
    for r in range(reps):
        i = block_idx(n, block, rng)
        (u1, s1), (u2, s2) = util_sharpe(a[i]), util_sharpe(b[i])
        bu[r], bs[r] = u1 - u2, s1 - s2
    return {"dUtility": d_u, "U_lo": np.percentile(bu, 2.5), "U_hi": np.percentile(bu, 97.5),
            "p_U": np.mean(np.abs(bu - d_u) >= abs(d_u)),
            "dSharpe": d_s, "S_lo": np.percentile(bs, 2.5), "S_hi": np.percentile(bs, 97.5),
            "p_S": np.mean(np.abs(bs - d_s) >= abs(d_s))}


def holm(p):
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def run_tests(sims, v):
    a, b = PERIODS[TEST_PERIOD]
    specs = [k for k in v if k != "Lagged RV"]
    models = sorted({k.rsplit("_", 1)[0] for k in specs})
    pairs = [("primary", f"{m}_dec", f"{m}_stat") for m in models
             if f"{m}_dec" in specs and f"{m}_stat" in specs]
    pairs += [("secondary", s, "Lagged RV") for s in specs]
    pairs += [("secondary", s, "Unmanaged") for s in specs]
    pairs += [("secondary", f"{m}_stat", "garch_stat") for m in models
              if m != "garch" and f"{m}_stat" in specs and "garch_stat" in specs]
    pairs += [("context", "Lagged RV", "Unmanaged")]
    rows = []
    for kind, A, B in pairs:
        x = sims[A].loc[a:b, "net_ex"].to_numpy()
        y = sims[B].loc[a:b, "net_ex"].to_numpy()
        rows.append({"kind": kind, "A": A, "B": B, **boot_diff(x, y)})
    out = pd.DataFrame(rows)
    prim = out.kind == "primary"
    if prim.any():
        out.loc[prim, "p_U_holm"] = holm(out.loc[prim, "p_U"].to_numpy())
    return out


def main():
    pd.set_option("display.width", 220)
    base, rf, v = load()
    print("forecasters:", list(v))

    acc = accuracy(base, v)
    print("\nFORECAST ACCURACY (QLIKE, MSE, bias = mean forecast / mean RV)")
    print(acc.round(4).to_string())

    c, w = make_weights(v)
    print("\nc fitted on 1970-1999, w_max = 2:", {k: round(x, 6) for k, x in c.items()})

    table, sims = backtest(base, rf, w)
    show = ["Sharpe", "Utility", "CAGR", "MaxDD", "AvgW", "AtCap", "Turnover"]
    for label in PERIODS:
        print(f"\nBACKTEST {label}, {BASE_KAPPA * 1e4:.0f} bps, gamma = {GAMMA:g}")
        print(table.xs(label, level="period")[show].round(4).to_string())

    tests = run_tests(sims, v)
    print(f"\nPAIRED BLOCK BOOTSTRAP, {TEST_PERIOD}, block = {BLOCK} months, {REPS} resamples"
          f" (A minus B; primary metric = net utility)")
    print(tests.round(4).to_string(index=False))
    if not (tests.kind == "primary").any():
        print("no primary pairs yet: needs both <model>_stat and <model>_dec forecast files")

    acc.to_csv(RES / "eval_accuracy.csv")
    table.to_csv(RES / "eval_backtest.csv")
    tests.to_csv(RES / "eval_tests.csv", index=False)


if __name__ == "__main__":
    main()
