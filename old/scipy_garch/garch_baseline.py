from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import arch
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import brentq


@dataclass(frozen=True)
class Config:
    series: str = "Mkt-RF"
    first_forecast_month: str = "1970-01"
    last_forecast_month: str = "2025-12"
    eval_start: str = "2000-01"
    horizon: int = 21
    min_observations: int = 1260


def load_data(daily_path, monthly_path, series="Mkt-RF"):
    """Accept the supplied long CSV/Parquet files or bundled market extracts."""
    def read(path):
        path = Path(path)
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    daily, monthly = read(daily_path), read(monthly_path)
    for frame, required in [(daily, {"Date", "series", "ret"}),
                            (monthly, {"series", "month", "ret_m", "rv", "rv_lag1", "n_days"})]:
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing columns: {required - set(frame.columns)}")
    daily = daily.loc[daily.series == series].copy()
    monthly = monthly.loc[monthly.series == series].copy()
    daily["Date"] = pd.to_datetime(daily.Date)
    monthly["month"] = pd.PeriodIndex(monthly.month.astype(str), freq="M")
    daily = daily.sort_values("Date").set_index("Date")
    monthly = monthly.sort_values("month").set_index("month")
    if daily.empty or monthly.empty or daily.index.has_duplicates or monthly.index.has_duplicates:
        raise ValueError("Empty series or duplicate dates/months")
    if not np.isfinite(daily.ret).all():
        raise ValueError("Daily returns must be finite")
    if not np.isfinite(monthly[["ret_m", "rv", "n_days"]].to_numpy()).all():
        raise ValueError("Monthly inputs must be finite")
    if not (monthly.rv > 0).all():
        raise ValueError("RV must be positive")
    expected = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    if not monthly.index.equals(expected.rename("month")):
        raise ValueError("Missing calendar month")
    rv = daily.ret.pow(2).groupby(daily.index.to_period("M")).sum()
    count = daily.ret.groupby(daily.index.to_period("M")).size()
    np.testing.assert_allclose(rv.reindex(monthly.index), monthly.rv, rtol=1e-10, atol=1e-14)
    np.testing.assert_array_equal(count.reindex(monthly.index), monthly.n_days)
    np.testing.assert_allclose(monthly.rv.shift(), monthly.rv_lag1, equal_nan=True,
                               rtol=1e-10, atol=1e-14)
    return daily, monthly


def history_at_origin(daily, target_month):
    """Return only observations strictly before the forecast target month."""
    target = pd.Period(target_month, freq="M")
    history = daily.loc[daily.index < target.start_time, "ret"]
    if history.empty or history.index.max().to_period("M") != target - 1:
        raise ValueError("History must end in the preceding month")
    return history


def fit_garch(history, start_params=None):
    model = arch_model(history * 100.0, mean="Zero", vol="GARCH", p=1, o=0,
                       q=1, dist="normal", rescale=False)
    attempts, notes = [], []
    starts = [start_params, None] if start_params is not None else [None, None]
    for attempt, start in enumerate(starts):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.fit(disp="off", update_freq=0, cov_type="robust",
                               starting_values=start,
                               options={"maxiter": 2000 if attempt == 0 else 5000,
                                        "ftol": 1e-8})
        notes.extend(str(item.message) for item in caught)
        attempts.append(int(result.convergence_flag))
        p = result.params
        valid = (np.isfinite(p).all() and p["omega"] > 0 and
                 p["alpha[1]"] >= 0 and p["beta[1]"] >= 0 and
                 p["alpha[1]"] + p["beta[1]"] <= 1.0 + 1e-8)
        if result.convergence_flag == 0 and valid:
            return result, len(attempts), " | ".join(notes)
    raise RuntimeError(f"GARCH fit failed at {history.index.max()}: {attempts}; {notes}")


def variance_path(result, horizon):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    p = result.params
    omega, alpha, beta = p["omega"], p["alpha[1]"], p["beta[1]"]
    last_h = float(np.asarray(result.conditional_volatility)[-1]) ** 2
    last_eps = float(np.asarray(result.resid)[-1])
    recursive = np.empty(horizon)
    recursive[0] = omega + alpha * last_eps ** 2 + beta * last_h
    for j in range(1, horizon):
        recursive[j] = omega + (alpha + beta) * recursive[j-1]
    packaged = result.forecast(horizon=horizon, method="analytic", reindex=False)
    packaged = packaged.residual_variance.iloc[-1].to_numpy(dtype=float)
    np.testing.assert_allclose(recursive, packaged, rtol=1e-9, atol=1e-10)
    if not np.isfinite(packaged).all() or not (packaged > 0).all():
        raise ValueError("Invalid variance forecast")
    return packaged / 10000.0


def forecast_one(history, horizon=21, start_params=None):
    """This function receives no future RV, returns or realized trading-day count."""
    result, attempts, messages = fit_garch(history, start_params)
    path = variance_path(result, horizon)
    params = result.params
    persistence = float(params["alpha[1]"] + params["beta[1]"])
    record = {
        "origin_date": history.index.max().strftime("%Y-%m-%d"),
        "training_start": history.index.min().strftime("%Y-%m-%d"),
        "n_train": len(history), "horizon": horizon,
        "garch_rv_forecast": float(path.sum()),
        "next_day_variance": float(path[0]),
        "omega_decimal": float(params["omega"] / 10000.0),
        "alpha": float(params["alpha[1]"]), "beta": float(params["beta[1]"]),
        "persistence": persistence,
        "unconditional_daily_variance": float(params["omega"] / 10000.0 / (1-persistence))
        if persistence < 1 - 1e-8 else None,
        "near_integrated": bool(persistence >= .999),
        "convergence_flag": int(result.convergence_flag),
        "fit_attempts": attempts, "fit_warnings": messages,
        "loglikelihood": float(result.loglikelihood),
    }
    return record, result


def walk_forward(daily, monthly, config=Config(), progress=False):
    """Expanding history; refit monthly; attach outcomes only after forecasting."""
    if config.horizon < 1:
        raise ValueError("Horizon must be positive")
    targets = pd.period_range(config.first_forecast_month, config.last_forecast_month, freq="M")
    if len(targets) == 0 or not targets.isin(monthly.index).all():
        raise ValueError("Requested target months unavailable")
    rows, start_params = [], None
    for i, target in enumerate(targets):
        history = history_at_origin(daily, target)
        if len(history) < config.min_observations:
            raise ValueError(f"Insufficient history for {target}")
        row, result = forecast_one(history, config.horizon, start_params)
        row["month"] = str(target)
        row["rv_baseline_forecast"] = float(history.loc[
            history.index.to_period("M") == target-1].pow(2).sum())
        start_params = result.params.to_numpy()
        rows.append(row)
        if progress and (i % 12 == 0 or i == len(targets)-1):
            print(f"{target}: {i+1}/{len(targets)} forecasts", flush=True)
    forecasts = pd.DataFrame(rows).set_index("month")
    actuals = monthly[["rv", "ret_m", "n_days", "rv_lag1"]].copy()
    actuals.index = actuals.index.astype(str)
    out = forecasts.join(actuals, validate="one_to_one")
    np.testing.assert_allclose(out.rv_baseline_forecast, out.rv_lag1, rtol=1e-10, atol=1e-14)
    origin_month = pd.to_datetime(out.origin_date).dt.to_period("M").astype(str)
    assert (origin_month.to_numpy() < out.index.to_numpy()).all()
    return out


def score_forecasts(forecasts):
    """QLIKE/MSE use the same months for each model. Lower is better."""
    actual = forecasts.rv.to_numpy()
    if not np.isfinite(actual).all() or not (actual > 0).all():
        raise ValueError("Invalid realized RV")
    records = []
    for label, column in [("Lagged RV", "rv_baseline_forecast"),
                           ("GARCH(1,1)", "garch_rv_forecast")]:
        predicted = forecasts[column].to_numpy()
        if not np.isfinite(predicted).all() or not (predicted > 0).all():
            raise ValueError("Invalid predicted RV")
        ratio = actual / predicted
        records.append({"model": label, "months": len(actual),
                        "QLIKE": float(np.mean(ratio - np.log(ratio) - 1)),
                        "MSE": float(np.mean((actual-predicted)**2)),
                        "mean_error_forecast_minus_RV": float(np.mean(predicted-actual)),
                        "mean_forecast_to_mean_RV": float(np.mean(predicted)/np.mean(actual))})
    return pd.DataFrame(records).set_index("model")


def plot_forecasts(forecasts, dark=True):

    forecasts = forecasts.sort_index()

    # Supports both a PeriodIndex and string/datetime month labels.
    dates = (
        forecasts.index.to_timestamp()
        if isinstance(forecasts.index, pd.PeriodIndex)
        else pd.to_datetime(forecasts.index)
    )

    horizon = int(forecasts["horizon"].iloc[0])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.08,
    )

    series = [
        ("rv", "Realized monthly RV",
         "#F1F5F9" if dark else "#263238"),
        ("rv_baseline_forecast", "Lagged RV forecast", "#F59E0B"),
        ("garch_rv_forecast",
         f"GARCH forecast ({horizon} days)", "#3B82F6"),
    ]

    for column, label, color in series:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=100 * np.sqrt(forecasts[column]),
                name=label,
                mode="lines",
                line=dict(color=color, width=2),
                hovertemplate=(
                    "%{x|%b %Y}<br>"
                    "√Variance: %{y:.2f}%"
                    "<extra>%{fullData.name}</extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=forecasts["persistence"],
            name="Persistence (α + β)",
            mode="lines",
            line=dict(color="#3B82F6", width=2),
            hovertemplate=(
                "%{x|%b %Y}<br>"
                "Persistence: %{y:.5f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        title="Market volatility: walk-forward forecast comparison",
        height=750,
        hovermode="x unified",
        dragmode="zoom",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=85, r=30, t=110, b=65),
    )

    fig.update_yaxes(
        title_text="√Monthly variance (%)",
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Persistence (α + β)",
        tickformat=".4f",
        row=2, col=1,
    )
    fig.update_xaxes(
        title_text="Forecast target month",
        row=2, col=1,
    )

    return fig


def save_run(forecasts, scores, config, daily_path, monthly_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output / "market_forecasts.csv")
    scores.to_csv(output / "forecast_metrics.csv")
    manifest = {"config": asdict(config), "purpose": "development, not final holdout",
                "mean_model": "zero", "distribution": "Gaussian QMLE",
                "window": "expanding; monthly re-estimation", "return_unit": "decimal",
                "forecast_unit": "decimal squared, sum over fixed horizon",
                "versions": {"python": platform.python_version(), "arch": arch.__version__,
                             "numpy": np.__version__, "pandas": pd.__version__},
                "inputs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                           for p in [daily_path, monthly_path]}}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

def main():
    root = Path(__file__).resolve().parents[2]   # project root (archived in old/scipy_garch)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
    "--daily",
    type=Path,
    default=root / "data/ff_daily_garchdata_long.parquet",)
    parser.add_argument(
    "--monthly",
    type=Path,
    default=root / "data/ff_monthly_baselinedata.parquet",)
    parser.add_argument("--start", default="1970-01")
    parser.add_argument("--end", default="2025-12")
    parser.add_argument("--eval-start", default="2000-01")
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--output", type=Path, default=root / "results_1970")
    args = parser.parse_args()
    config = Config(first_forecast_month=args.start, last_forecast_month=args.end, eval_start = args.eval_start,
                    horizon=args.horizon)
    daily, monthly = load_data(args.daily, args.monthly, config.series)
    forecasts = walk_forward(daily, monthly, config, progress=True)
    scores = score_forecasts(forecasts.loc[config.eval_start:])
    save_run(forecasts, scores, config, args.daily, args.monthly, args.output)
    print(scores.to_string())
    fig = plot_forecasts(forecasts)
    html_path = args.output / "forecast_comparison.html"
    fig.write_html(html_path,
    include_plotlyjs=True,
    config={"scrollZoom": True, "displaylogo": False},
    auto_open=False,)

    print(f"Interactive plot saved to: {html_path.resolve()}")


if __name__ == "__main__":
    main()
