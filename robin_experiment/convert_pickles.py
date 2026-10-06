"""Converts Robin's saved backtest panels (results/bt_<factor>.pkl and results/zhao/bt_<factor>.pkl, written by backtest.py and
backtest_zhao.py) into the plain CSV files ../data/robin/<factor>_forecasts.csv that ../9_dfl_robin.py reads.
Columns: f (factor excess return of the month), rv (realised variance of the month), V_<model> (real-time variance forecast for
the month, made at the end of the previous month) and c_<model> (Robin's scaling constant for that year).
The pickles are not in the repository (they are re-created by backtest.py / backtest_zhao.py, about 1.5 h on 2 cores).
A restricted unpickler is used so that only pandas / numpy objects can be loaded."""
import pickle
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "robin"


class Safe(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split(".")[0] not in ("numpy", "pandas", "builtins") or (mod == "builtins" and name not in ("slice", "object", "int", "float", "list", "dict", "tuple", "set", "frozenset", "complex", "range")):
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        return super().find_class(mod, name)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fac in ["Mkt", "SMB", "HML", "Mom", "RMW", "CMA"]:
        g = Safe(open(HERE / "results" / f"bt_{fac}.pkl", "rb")).load()["months"].set_index("month").sort_index()
        z = Safe(open(HERE / "results" / "zhao" / f"bt_{fac}.pkl", "rb")).load()["months"].set_index("month").sort_index()
        df = g[["f", "rv"] + [c for c in g.columns if c.startswith(("V_", "c_"))]].join(z[[c for c in z.columns if c.startswith(("V_zhao", "c_zhao"))]])
        df.index = df.index.astype(str); df.index.name = "month"
        df.to_csv(OUT / f"{fac}_forecasts.csv", float_format="%.10g"); print(fac, df.shape)
