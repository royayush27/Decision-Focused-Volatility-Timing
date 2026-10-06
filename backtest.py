from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results_1970")
GAMMA = 3.0
BASE_KAPPA = 0.0005
KAPPAS = [0.0, 0.0005, 0.0010, 0.0025]
PERIODS = {"dev 2000-2024": ("2000-01", "2024-12"), "check 2025": ("2025-01", "2025-12")}
CAP = 2.0


def simulate(w, ex, rf, kappa):
    w, ex, rf = w.to_numpy(float), ex.to_numpy(float), rf.to_numpy(float)
    R = ex + rf
    rp = rf + w * ex
    drift = np.empty_like(w)
    drift[0] = w[0]
    drift[1:] = w[:-1] * (1 + R[:-1]) / (1 + rp[:-1])
    turnover = np.abs(w - drift)
    cost = kappa * turnover
    net_ex = w * ex - cost
    return pd.DataFrame({"w": w, "net_ex": net_ex, "net_tot": rf + net_ex,
                         "turnover": turnover, "cost": cost})


def metrics(sim, gamma=GAMMA):
    n = len(sim)
    mu, var = sim.net_ex.mean() * 12, sim.net_ex.var(ddof=1) * 12
    wealth = (1 + sim.net_tot).cumprod()
    peak = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return pd.Series({
        "Sharpe": mu / np.sqrt(var),
        "CAGR": wealth.iloc[-1] ** (12 / n) - 1,
        "MaxDD": (wealth / peak - 1).min(),
        "Utility": mu - gamma / 2 * var,
        "Vol": np.sqrt(var),
        "AvgW": sim.w.mean(),
        "AtCap": (sim.w >= CAP - 1e-12).mean(),
        "Turnover": sim.turnover.mean(),
        "CostDrag": sim.cost.mean() * 12,
    })


def run(fc, weights, rf, kappa):
    ex = fc["ret_m"]
    out = {}
    for name in ["Unmanaged", "Lagged RV", "GARCH(1,1)"]:
        w = pd.Series(1.0, index=fc.index) if name == "Unmanaged" else weights[name]
        sim = simulate(w, ex, rf, kappa)
        sim.index = fc.index
        out[name] = sim
    return out


def main():
    fc = pd.read_csv(RES / "market_forecasts.csv", index_col="month")
    weights = pd.read_csv(RES / "weights.csv", index_col="month")
    rf = pd.read_csv("data/ff_monthly_rf.csv", index_col="month")["RF"].reindex(fc.index)
    assert rf.notna().all() and weights.index.equals(fc.index)

    rows = []
    for kappa in KAPPAS:
        sims = run(fc, weights, rf, kappa)
        for label, (a, b) in PERIODS.items():
            ex, r = fc["ret_m"].loc[a:b], rf.loc[a:b]
            for name, sim in sims.items():
                rows.append(metrics(sim.loc[a:b]).rename((kappa, label, name)))
            for name in ["Lagged RV", "GARCH(1,1)"]:
                k = sims[name].loc[a:b, "w"].mean()
                sim = simulate(pd.Series(k, index=ex.index), ex, r, kappa)
                sim.index = ex.index
                rows.append(metrics(sim).rename((kappa, label, f"Static {k:.2f} (= {name} avg)")))
    table = pd.DataFrame(rows)
    table.index.names = ["kappa", "period", "strategy"]
    table.to_csv(RES / "backtest_metrics.csv")

    pd.set_option("display.width", 200)
    for label in PERIODS:
        base = table.xs((BASE_KAPPA, label), level=("kappa", "period"))
        print(f"\n{label}, cost {BASE_KAPPA * 1e4:.0f} bps per unit turnover")
        print(base.round(4).to_string())
    print("\nSharpe by cost level, dev 2000-2024")
    print(table.xs("dev 2000-2024", level="period")["Sharpe"].unstack(0).round(3).to_string())


if __name__ == "__main__":
    main()
