import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

W_MAX = 2.0
CAL_END = "1999-12"
MODELS = {"Lagged RV": "rv_baseline_forecast", "GARCH(1,1)": "garch_rv_forecast"}


def exposure(v, c, w_max=W_MAX):
    return np.clip(c / np.asarray(v, dtype=float), 0, w_max)


def solve_c(v, w_max=W_MAX, target=1.0):
    v = np.asarray(v, dtype=float)
    if not (np.isfinite(v).all() and (v > 0).all()):
        raise ValueError("forecasts must be finite and positive")
    if w_max <= target:
        raise ValueError("w_max must be above the target exposure")
    f = lambda c: exposure(v, c, w_max).mean() - target
    lo = 0.5 / np.mean(1 / v)
    hi = w_max * v.max()
    return brentq(f, lo, hi, xtol=1e-14)


def calibrate(fc, cal_end=CAL_END, w_max=W_MAX):
    cal = fc.loc[:cal_end]
    c = {m: solve_c(cal[col], w_max) for m, col in MODELS.items()}
    w = pd.DataFrame({m: exposure(fc[col], c[m], w_max) for m, col in MODELS.items()},
                     index=fc.index)
    return c, w


def main():
    path = Path("results_1970") / "market_forecasts.csv"
    fc = pd.read_csv(path, index_col="month")
    assert fc.index.is_monotonic_increasing and fc.index[0] == "1970-01"

    c, w = calibrate(fc)
    for m in MODELS:
        cal, ev = w.loc[:CAL_END, m], w.loc["2000-01":, m]
        assert abs(cal.mean() - 1) < 1e-8
        print(f"{m:11s} c = {c[m]:.6f} | cal mean w = {cal.mean():.4f}, at cap = {(cal >= W_MAX).mean():.1%}"
              f" | eval mean w = {ev.mean():.4f}, at cap = {(ev >= W_MAX).mean():.1%}")

    w.to_csv(path.with_name("weights.csv"))
    path.with_name("c_values.json").write_text(
        json.dumps({"w_max": W_MAX, "cal_end": CAL_END, "c": c}, indent=2) + "\n")


if __name__ == "__main__":
    main()
