"""Logging and saving: every script prints to the screen AND to results/<spec>/<script>.txt, and saves its monthly net
returns to results/<spec>/nets_<script>.csv (used by 8_multiple_testing.py)."""
import sys

import pandas as pd

import config


class _Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8"); self.out = sys.stdout

    def write(self, s):
        self.out.write(s); self.f.write(s)

    def flush(self):
        self.out.flush(); self.f.flush()


def start(name, title):
    s = config.spec()
    sys.stdout = _Tee(config.out_dir() / f"{name}.txt")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 50)
    cap = "none" if s["cap"] is None else s["cap"]
    print(f"{title}\nspec '{s['name']}': cost {1e4 * s['cost']:.0f} bp per unit of exposure change, cap {cap}, c rule {s['c_rule']}, "
          f"bootstrap {config.BLOCK}-month blocks x {config.N_BOOT}\n")
    return s


def save_nets(name, nets, start="1998-01", end="2025-12"):
    """nets: {(group, series, rule): Series of monthly net excess returns}. 'Unmanaged' must be one of the rules."""
    df = pd.DataFrame({" | ".join(k): v.loc[start:end] for k, v in nets.items()})
    df.index.name = "month"; df.to_csv(config.out_dir() / f"nets_{name}.csv")


def show(T, nd=2):
    """Print a table with numeric columns rounded."""
    T = pd.DataFrame(T).copy()
    for c in T.columns:
        x = pd.to_numeric(T[c], errors="coerce")
        if x.notna().sum() == T[c].notna().sum():
            T[c] = x.round(nd)
    print(T.to_string(), "\n")
