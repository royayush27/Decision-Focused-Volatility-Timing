"""Real-time (Cederburg et al. 2020 style) expanding-window backtest.

For every factor and every year Y from the first year with >= 10 years of daily data:
  * fit each model on all daily data before 1 Jan Y  (objective A, then objective B starting from A)
  * with parameters frozen, produce the month-ahead variance forecast at each month-end of Y-1..Y
    (the filter only uses data up to the forecast date, so nothing is looked ahead)
  * set the scaling constant c on the training months only (managed sd = unmanaged sd in-sample)
The Moreira-Muir rule (V = last month's realized variance) is run the same way, with a real-time c.

Usage: python backtest.py FACTOR [FACTOR ...]
"""
import sys, time, pickle, copy
import numpy as np
import pandas as pd
import torch
from data import load
from models import VolModel
from train import fit_A_garch, fit_A_lstm, fit_B
from mm_replication import realized_variance

torch.set_num_threads(1)
import os
LAST_YEAR = int(os.environ.get("LAST_YEAR", 2026))
SEED_OFFSET = int(os.environ.get("SEED_OFFSET", 0))


def c_scale(V, f):
    raw = f / V
    return float(np.std(f, ddof=1) / np.std(raw, ddof=1))


def run(fac):
    D, M = load(True), load(False)
    d = D[fac].dropna()
    r_all = torch.tensor(d.values)
    per = d.index.to_period("M")
    me = pd.Series(np.arange(len(d)), index=d.index).groupby(per).max()
    rv = realized_variance(d.to_frame())[fac]
    months = pd.DataFrame({"me_prev": me.shift(1), "f": M[fac].reindex(me.index),
                           "rv": rv, "rv_prev": rv.shift(1)}).dropna()
    months["me_prev"] = months["me_prev"].astype(int)
    first_year = d.index[0].year + 11 if d.index[0].month > 1 else d.index[0].year + 10
    years = list(range(first_year, LAST_YEAR + 1))
    me_idx_all = torch.tensor(months.me_prev.values)
    f_all = months.f.values

    specs = [("garch", "A"), ("garch", "B"), ("gjr", "A"), ("gjr", "B"), ("lstm", "A"), ("lstm", "B")]
    if SEED_OFFSET:   # robustness run: only the network changes with the seed
        specs = [("garch", "A"), ("gjr", "A"), ("lstm", "A"), ("lstm", "B")]
    prev = {}
    rows, params = [], []
    for Y in years:
        t0 = time.time()
        cut = pd.Timestamp(f"{Y}-01-01")
        n_tr = int((d.index < cut).sum())
        r_tr = r_all[:n_tr]
        tr_m = months.index.to_timestamp() < cut
        te_m = months.index.year == Y
        if te_m.sum() == 0:
            continue
        me_tr = me_idx_all[tr_m]
        f_tr = torch.tensor(f_all[tr_m])
        var0 = float(r_tr.var())
        out = {"RV": months.rv_prev.values}
        c_out = {"RV": c_scale(months.rv_prev.values[tr_m], f_all[tr_m])}
        for kind, obj in specs:
            torch.manual_seed(Y + SEED_OFFSET)
            if obj == "A":
                m = VolModel(kind, var0)
                if kind == "lstm":
                    init = prev.get(("lstm", "A"))
                    if init is not None:   # warm start the network weights only, from last year's fit
                        m.load_state_dict(init.state_dict())
                    # the GJR block always starts from this year's fresh GJR QMLE fit
                    m.load_state_dict({**m.state_dict(), **models[("gjr", "A")].state_dict()})
                    fit_A_lstm(m, r_tr, seed=Y + SEED_OFFSET)
                else:
                    fit_A_garch(m, r_tr)   # fresh start every year (no warm start: avoids sticking at bounds)
            else:
                m = copy.deepcopy(models[(kind, "A")])
                fit_B(m, r_tr, me_tr, f_tr, lr=(3e-3 if kind == "lstm" else 1e-2))
            if obj == "A" and kind == "garch":
                models = {}
            models[(kind, obj)] = m
            with torch.no_grad():
                V = m.monthly_forecast(r_all, me_idx_all).numpy()
            key = f"{kind}_{obj}"
            out[key] = V
            c_out[key] = c_scale(V[tr_m], f_all[tr_m])
            params.append({"factor": fac, "year": Y, "model": key, **m.readable()})
        prev = models
        for i in np.where(te_m)[0]:
            row = {"factor": fac, "month": months.index[i], "f": f_all[i], "rv": months.rv.values[i]}
            for k in out:
                row[f"V_{k}"] = out[k][i]
                row[f"c_{k}"] = c_out[k]
            rows.append(row)
        print(fac, Y, f"{time.time()-t0:.1f}s", flush=True)
    res = {"months": pd.DataFrame(rows), "params": pd.DataFrame(params)}
    suffix = f"_seed{SEED_OFFSET}" if SEED_OFFSET else ""
    with open(f"results/{'seeds/' if SEED_OFFSET else ''}bt_{fac}{suffix}.pkl", "wb") as fh:
        pickle.dump(res, fh)


if __name__ == "__main__":
    for fac in sys.argv[1:]:
        run(fac)
