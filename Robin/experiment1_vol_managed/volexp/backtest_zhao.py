"""Real-time backtest of the Zhao et al. (2024) GARCH-LSTM, 5 seeds per refit, objectives A and B.

Same protocol as backtest.py: refit every January on data before that date, forecasts at each month-end,
c set on past months only. For each seed the network warm-starts from that seed's fit of the previous
year; the GJR block always starts from this year's fresh GJR QMLE fit.

Usage: python backtest_zhao.py FACTOR [FACTOR ...]
"""
import os, sys, time, pickle, copy
import numpy as np
import pandas as pd
import torch
from data import load
from models import VolModel
from train import fit_A_garch, fit_A_zhao, fit_B
from mm_replication import realized_variance
from backtest import c_scale

torch.set_num_threads(1)
N_SEEDS = int(os.environ.get("N_SEEDS", 5))
LAST_YEAR = int(os.environ.get("LAST_YEAR", 2026))


def run(fac):
    D, M = load(True), load(False)
    d = D[fac].dropna()
    r_all = torch.tensor(d.values)
    per = d.index.to_period("M")
    me = pd.Series(np.arange(len(d)), index=d.index).groupby(per).max()
    rv = realized_variance(d.to_frame())[fac]
    months = pd.DataFrame({"me_prev": me.shift(1), "f": M[fac].reindex(me.index), "rv": rv}).dropna()
    months["me_prev"] = months["me_prev"].astype(int)
    first_year = d.index[0].year + 11 if d.index[0].month > 1 else d.index[0].year + 10
    me_idx_all = torch.tensor(months.me_prev.values)
    f_all = months.f.values
    usable = months.me_prev.values >= 125           # months whose forecast has a full 126-day window
    prev = {}
    rows, params = [], []
    for Y in range(first_year, LAST_YEAR + 1):
        t0 = time.time()
        cut = pd.Timestamp(f"{Y}-01-01")
        n_tr = int((d.index < cut).sum())
        r_tr = r_all[:n_tr]
        tr_m = (months.index.to_timestamp() < cut)
        te_m = months.index.year == Y
        if te_m.sum() == 0:
            continue
        var0 = float(r_tr.var())
        gjr = VolModel("gjr", var0)
        fit_A_garch(gjr, r_tr)
        b_m = tr_m & usable
        me_tr, f_tr = me_idx_all[b_m], torch.tensor(f_all[b_m])
        out, c_out = {}, {}
        for s in range(N_SEEDS):
            seed = 100 * s + Y
            torch.manual_seed(seed)
            m = VolModel("zhao", var0)
            if s in prev:
                m.load_state_dict(prev[s].state_dict())   # warm start the network
            m.load_state_dict({**m.state_dict(), **gjr.state_dict()})
            fit_A_zhao(m, r_tr, seed=seed)
            prev[s] = copy.deepcopy(m)
            mB = copy.deepcopy(m)
            fit_B(mB, r_tr, me_tr, f_tr, lr=3e-3)
            for obj, mm in (("A", m), ("B", mB)):
                with torch.no_grad():
                    V = mm.monthly_forecast(r_all, me_idx_all).numpy()
                key = f"zhao_{obj}_s{s}"
                out[key] = V
                c_out[key] = c_scale(V[tr_m], f_all[tr_m])
                params.append({"factor": fac, "year": Y, "model": key, **mm.readable(),
                               "w_norm": float(mm.u_out.detach().abs().sum() / (1 + mm.u_out.detach().abs().sum()))})
        for i in np.where(te_m)[0]:
            row = {"factor": fac, "month": months.index[i], "f": f_all[i], "rv": months.rv.values[i]}
            for k in out:
                row[f"V_{k}"] = out[k][i]
                row[f"c_{k}"] = c_out[k]
            rows.append(row)
        print(fac, Y, f"{time.time()-t0:.1f}s", flush=True)
    os.makedirs("results/zhao", exist_ok=True)
    with open(f"results/zhao/bt_{fac}.pkl", "wb") as fh:
        pickle.dump({"months": pd.DataFrame(rows), "params": pd.DataFrame(params)}, fh)


if __name__ == "__main__":
    for fac in sys.argv[1:]:
        run(fac)
