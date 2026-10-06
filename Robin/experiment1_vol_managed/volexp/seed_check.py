"""Robustness of the GARCH-LSTM results to the network's random seed."""
import glob, pickle
import numpy as np
import pandas as pd
from evaluate import block_boot_dsr, sharpe, KAPPA


def lstm_returns(months, key):
    g = months.set_index("month").sort_index()
    w = g[f"c_{key}"] / g[f"V_{key}"]
    return w, w * g["f"], g["f"]


rows = []
for p in sorted(glob.glob("results/seeds/bt_*_seed*.pkl")):
    fac = p.split("bt_")[1].split("_seed")[0]
    s1 = pickle.load(open(p, "rb"))["months"]
    s0 = pickle.load(open(f"results/bt_{fac}.pkl", "rb"))["months"]
    out = {}
    for tag, m in [("seed0", s0), ("seed1", s1)]:
        for obj in "AB":
            w, R, f = lstm_returns(m, f"lstm_{obj}")
            out[(tag, obj)] = (w, R)
    for obj in "AB":   # two-seed ensemble: average the two weights
        w = (out[("seed0", obj)][0] + out[("seed1", obj)][0]) / 2
        out[("ens", obj)] = (w, w * f)
    r = {"factor": fac}
    for tag in ["seed0", "seed1", "ens"]:
        RA, RB = out[(tag, "A")][1], out[(tag, "B")][1]
        d, se, pv = block_boot_dsr(RB.values, RA.values)
        r.update({f"{tag}_SR_A": sharpe(RA), f"{tag}_SR_B": sharpe(RB), f"{tag}_dSR": d, f"{tag}_p": pv})
    r["BH"] = sharpe(f)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv("results/seed_check.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
