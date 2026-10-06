# archived in old/scipy_garch: run from anywhere, paths below are relative to the project root
import os as _os, sys as _sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[2]
for _p in (_ROOT, _P(__file__).resolve().parent):
    _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)

from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

A, B = "gjr", "gjr_t"
PERIODS = {"cal 1970-99": ("1970-01", "1999-12"), "dev 2000-24": ("2000-01", "2024-12"),
           "2025": ("2025-01", "2025-12")}
FC = Path("results_1970/forecasts")


def qlike(rv, v):
    r = rv / v
    return r - np.log(r) - 1


def dm(x, lags=3):
    x = np.asarray(x, float)
    n, e = len(x), x - x.mean()
    v = e @ e / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    t = x.mean() / np.sqrt(v / n)
    return t, 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def main():
    rv = pd.read_csv("results_1970/market_forecasts.csv", index_col="month")["rv"]
    v = {k: pd.read_csv(FC / f"{k}_stat.csv", index_col="month")["v_hat"] for k in (A, B)}
    diag = {k: pd.read_csv(FC / f"{k}_stat_diag.csv", index_col="month") for k in (A, B)}
    assert v[A].index.equals(v[B].index) and v[A].index.equals(rv.index)

    rows = []
    for label, (a, b) in PERIODS.items():
        for k in (A, B):
            r, f = rv.loc[a:b], v[k].loc[a:b]
            rows.append({"period": label, "spec": k, "QLIKE": qlike(r, f).mean(),
                         "MSE_e5": ((r - f) ** 2).mean() * 1e5, "bias": f.mean() / r.mean()})
        d = qlike(rv.loc[a:b], v[B].loc[a:b]) - qlike(rv.loc[a:b], v[A].loc[a:b])
        t, p = dm(d)
        rows.append({"period": label, "spec": f"{B} - {A}", "QLIKE": d.mean(), "DM_t": t, "DM_p": p})
    print(pd.DataFrame(rows).set_index(["period", "spec"]).round(4).to_string())

    cal = slice("1970-01", "1999-12")
    stab = {}
    for k in (A, B):
        g = diag[k].loc[cal]
        stab[k] = {"alpha_std": g.alpha.std(), "beta_std": g.beta.std(), "gamma_std": g.gamma.std(),
                   "persist_min": g.persistence.min(), "persist_max": g.persistence.max(),
                   "persist_jump": g.persistence.diff().abs().mean(),
                   "near_integrated": int(g.near_integrated.sum())}
        if "nu" in g:
            stab[k].update({"nu_min": g.nu.min(), "nu_median": g.nu.median(), "nu_max": g.nu.max()})
    print("\nparameter stability, calibration years 1970-99")
    print(pd.DataFrame(stab).round(4).to_string())


if __name__ == "__main__":
    main()
