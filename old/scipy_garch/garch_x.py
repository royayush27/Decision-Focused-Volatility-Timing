import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# GARCH-X: h_t = omega + alpha*r_{t-1}^2 + beta*h_{t-1} + delta*x_{t-1}
# r in percent, x = VIX^2/252 (percent^2 per day). delta = 0 gives plain GARCH(1,1).


def load_inputs(daily_csv, vix_csv, series="Mkt-RF", first="1990-01-01"):
    vix = pd.read_csv(vix_csv)
    vix.columns = ["date", "VIXCLS"]
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date")

    daily = pd.read_csv(daily_csv)
    mkt = daily[daily["series"] == series].copy()
    mkt["Date"] = pd.to_datetime(mkt["Date"], format="%d-%m-%Y")
    r = mkt.set_index("Date")["ret"].sort_index().loc[first:]

    v = vix["VIXCLS"].reindex(r.index).ffill(limit=3)
    assert v.isna().sum() == 0, "return days without VIX"
    return r * 100, v ** 2 / 252  # both as Series, percent units


def variance_path(theta, r, x):
    omega, alpha, beta, delta = theta
    n = len(r)
    h = np.empty(n)
    h[0] = np.var(r)
    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1] + delta * x[t - 1]
    return h


def negative_loglikelihood(theta, r, x):
    omega, alpha, beta, delta = theta
    if alpha + beta >= 0.999 or omega <= 0 or alpha < 0 or beta < 0 or delta < 0:
        return 1e10
    h = variance_path(theta, r, x)
    if (h <= 0).any():
        return 1e10

    log_like = (1 / 2) * np.sum(np.log(h[1:]) + r[1:] ** 2 / h[1:])
    return log_like


def simulate(theta, x, seed=0):
    omega, alpha, beta, delta = theta
    rng = np.random.default_rng(seed)
    n = len(x)
    z = rng.standard_normal(n)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = (omega + delta * x.mean()) / (1 - alpha - beta)
    r[0] = np.sqrt(h[0]) * z[0]

    for t in range(1, n):
        h[t] = omega + alpha * r[t - 1] ** 2 + beta * h[t - 1] + delta * x[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]

    return r, h


def fit(r, x, delta_free=True, warm=None):
    # tries the default start and, if given, the previous solution; keeps the lower NLL
    starts = [[0.05, 0.05, 0.90, 0.01 if delta_free else 0.0]]
    if warm is not None:
        starts.append(list(warm))
    bounds = [(1e-6, None), (0, 1), (0, 1), (0, None) if delta_free else (0, 0)]
    best = None
    for start in starts:
        res = minimize(fun=lambda p: negative_loglikelihood(p, r, x) / len(r), x0=start,
                       method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    return best


def forecast(theta, r, x, horizon=21):
    omega, alpha, beta, delta = theta
    h = variance_path(theta, r, x)
    path = np.empty(horizon)
    path[0] = omega + alpha * r[-1] ** 2 + beta * h[-1] + delta * x[-1]
    for k in range(1, horizon):
        path[k] = (omega + delta * x[-1]) + (alpha + beta) * path[k - 1]
    return path


def walk_forward(r, x, first, last, delta_free=True, min_obs=1260, horizon=21):
    # for each target month, fit on every day before that month and forecast the next 21 days
    rows, warm = [], None
    for target in pd.period_range(first, last, freq="M"):
        cutoff = target.start_time
        past = r.index < cutoff
        rh, xh = r[past].to_numpy(), x[past].to_numpy()
        if len(rh) < min_obs:
            raise ValueError(f"only {len(rh)} days before {target}")
        res = fit(rh, xh, delta_free, warm)
        warm = res.x
        path = forecast(res.x, rh, xh, horizon)
        rows.append({"month": str(target), "v_hat": path.sum() / 1e4,
                     "omega": res.x[0], "alpha": res.x[1], "beta": res.x[2], "delta": res.x[3],
                     "persistence": res.x[1] + res.x[2], "nll": res.fun,
                     "ok": res.success, "n_train": len(rh)})
        if target.month == 1:
            print(target, flush=True)
    return pd.DataFrame(rows).set_index("month")


def main():
    root = Path(__file__).resolve().parents[2]   # project root (archived in old/scipy_garch)
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["garchx", "garch"], required=True)
    p.add_argument("--daily", type=Path, default=root / "data/ff_daily_garchdata_long.csv")
    p.add_argument("--vix", type=Path, default=root / "data/VIXCLS.csv")
    p.add_argument("--start", default="1995-01")
    p.add_argument("--end", default="2025-12")
    p.add_argument("--out", type=Path, default=root / "old" / "scipy_garch" / "forecasts_vix")
    a = p.parse_args()

    r, x = load_inputs(a.daily, a.vix)
    out = walk_forward(r, x, a.start, a.end, delta_free=(a.model == "garchx"))

    a.out.mkdir(parents=True, exist_ok=True)
    name = "garchx_vix" if a.model == "garchx" else "garch_1990"
    out[["v_hat"]].to_csv(a.out / f"{name}_stat.csv")
    out.drop(columns=["v_hat"]).to_csv(a.out / f"{name}_stat_diag.csv")
    print("saved", a.out / f"{name}_stat.csv", len(out), "rows")


if __name__ == "__main__":
    main()
