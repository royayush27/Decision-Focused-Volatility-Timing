from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

from garch_baseline import Config, history_at_origin, load_data


def fit_stat(history, model, dist="normal", start_params=None):
    o = 1 if model == "gjr" else 0
    am = arch_model(history * 100.0, mean="Zero", vol="GARCH", p=1, o=o, q=1,
                    dist=dist, rescale=False)
    attempts, notes = [], []
    starts = [start_params, None] if start_params is not None else [None, None]
    for attempt, start in enumerate(starts):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = am.fit(disp="off", update_freq=0, cov_type="robust",
                            starting_values=start,
                            options={"maxiter": 2000 if attempt == 0 else 5000, "ftol": 1e-8})
        notes.extend(str(item.message) for item in caught)
        attempts.append(int(result.convergence_flag))
        p = result.params
        g = p["gamma[1]"] if model == "gjr" else 0.0
        valid = (np.isfinite(p).all() and p["omega"] > 0 and p["alpha[1]"] >= 0
                 and p["beta[1]"] >= 0 and p["alpha[1]"] + g >= 0
                 and p["alpha[1]"] + g / 2 + p["beta[1]"] <= 1.0 + 1e-8)
        if result.convergence_flag == 0 and valid:
            return result, len(attempts), " | ".join(notes)
    raise RuntimeError(f"{model} fit failed at {history.index.max()}: {attempts}; {notes}")


def variance_path(result, model, horizon):
    p = result.params
    omega, alpha, beta = p["omega"], p["alpha[1]"], p["beta[1]"]
    gamma = p["gamma[1]"] if model == "gjr" else 0.0
    persistence = alpha + gamma / 2 + beta
    last_h = float(np.asarray(result.conditional_volatility)[-1]) ** 2
    last_eps = float(np.asarray(result.resid)[-1])
    recursive = np.empty(horizon)
    recursive[0] = omega + (alpha + gamma * (last_eps < 0)) * last_eps ** 2 + beta * last_h
    for j in range(1, horizon):
        recursive[j] = omega + persistence * recursive[j - 1]
    packaged = result.forecast(horizon=horizon, method="analytic", reindex=False)
    packaged = packaged.residual_variance.iloc[-1].to_numpy(dtype=float)
    np.testing.assert_allclose(recursive, packaged, rtol=1e-9, atol=1e-10)
    if not (np.isfinite(recursive).all() and (recursive > 0).all()):
        raise ValueError("Invalid variance forecast")
    return recursive / 10000.0, persistence


def walk_forward_stat(daily, model, config, dist="normal"):
    targets = pd.period_range(config.first_forecast_month, config.last_forecast_month, freq="M")
    rows, start_params = [], None
    for i, target in enumerate(targets):
        history = history_at_origin(daily, target)
        if len(history) < config.min_observations:
            raise ValueError(f"Insufficient history for {target}")
        result, attempts, notes = fit_stat(history, model, dist, start_params)
        path, persistence = variance_path(result, model, config.horizon)
        start_params = result.params.to_numpy()
        row = {"month": str(target), "v_hat": float(path.sum()),
               "next_day_variance": float(path[0]), "persistence": float(persistence),
               "near_integrated": bool(persistence >= 0.999),
               "n_train": len(history), "fit_attempts": attempts, "fit_warnings": notes}
        for name, value in result.params.items():
            row[name.replace("[1]", "")] = float(value)
        rows.append(row)
        if i % 12 == 0 or i == len(targets) - 1:
            print(f"{model} {dist} {target}: {i + 1}/{len(targets)}", flush=True)
    return pd.DataFrame(rows).set_index("month")


def main():
    root = Path(__file__).resolve().parents[2]   # project root (archived in old/scipy_garch)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["garch", "gjr"], required=True)
    parser.add_argument("--dist", choices=["normal", "t", "ged"], default="normal")
    parser.add_argument("--daily", type=Path, default=root / "data/ff_daily_garchdata_long.parquet")
    parser.add_argument("--monthly", type=Path, default=root / "data/ff_monthly_baselinedata.parquet")
    parser.add_argument("--start", default="1970-01")
    parser.add_argument("--end", default="2025-12")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--out", type=Path, default=root / "results_1970" / "forecasts")
    parser.add_argument("--check-against", type=Path, default=None)
    args = parser.parse_args()

    config = Config(first_forecast_month=args.start, last_forecast_month=args.end,
                    horizon=args.horizon)
    daily, monthly = load_data(args.daily, args.monthly, config.series)
    out = walk_forward_stat(daily, args.model, config, args.dist)

    name = args.model if args.dist == "normal" else f"{args.model}_{args.dist}"
    args.out.mkdir(parents=True, exist_ok=True)
    out[["v_hat"]].to_csv(args.out / f"{name}_stat.csv")
    out.drop(columns=["v_hat"]).to_csv(args.out / f"{name}_stat_diag.csv")
    print(f"saved {args.out / (name + '_stat.csv')}  rows={len(out)}")
    print(out[["persistence"]].describe().loc[["min", "mean", "max"]].T.to_string())
    print("near-integrated months:", int(out.near_integrated.sum()))

    if args.check_against is not None:
        old = pd.read_csv(args.check_against, index_col="month")["garch_rv_forecast"]
        common = out.index.intersection(old.index)
        rel = (out.loc[common, "v_hat"] / old.loc[common] - 1).abs().max()
        print(f"regression vs {args.check_against.name}: {len(common)} months, max rel diff {rel:.2e}")


if __name__ == "__main__":
    main()
