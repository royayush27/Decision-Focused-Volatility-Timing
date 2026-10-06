"""Robin's decision-trained GARCH-LSTM (Zhao et al. 2024, 5-seed ensemble) re-read on OUR horizons.
Reads Robin/experiment1_vol_managed/volexp/results/zhao/bt_<factor>.pkl (his real-time forecasts V and scaling c, 1937-2026)
and reports net Sharpe (10 bp per unit of weight change, his cost) for buy-and-hold, objective A (QMLE) and objective B
(decision / utility training) over 1998-2025, 1998-2015, 2016-2025, 2021-2025, with a 12-month block-bootstrap CI for B - A.
Two versions: his rule (no leverage cap) and ours (w capped at 2). Nothing is re-estimated: this only re-slices his output.
Pickles are loaded with a restricted unpickler that only accepts numpy / pandas objects."""
from pathlib import Path as _P
import pickle
class R(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "builtins" and name not in ("slice", "complex", "set", "frozenset", "range", "object", "int", "float", "list", "dict", "tuple"):
            raise pickle.UnpicklingError(f"blocked builtins.{name}")
        if not (mod.split(".")[0] in ("numpy", "pandas", "builtins")):
            raise pickle.UnpicklingError(f"blocked {mod}.{name}")
        return super().find_class(mod, name)
def load(p): return R(open(p, "rb")).load()

import numpy as np, pandas as pd
B = str(_P(__file__).resolve().parents[1] / 'Robin/experiment1_vol_managed/volexp/results/zhao/bt_{}.pkl')
P = {"1998-2025": ("1998-01", "2025-12"), "1998-2015": ("1998-01", "2015-12"), "2016-2025": ("2016-01", "2025-12"), "2021-2025": ("2021-01", "2025-12")}
sr = lambda x: np.sqrt(12) * x.mean(-1) / x.std(-1, ddof=1)
rng = np.random.default_rng(0)
rows = []
for cap in (None, 2.0):
    for fac in ["Mkt", "SMB", "HML", "Mom", "RMW", "CMA"]:
        g = load(B.format(fac))["months"].set_index("month").sort_index()
        net = {"BH": g["f"]}
        for o in "AB":
            ks = sorted({c[2:] for c in g.columns if c.startswith(f"V_zhao_{o}_")})
            w = sum(g[f"c_{k}"] / g[f"V_{k}"] for k in ks) / len(ks)
            if cap: w = w.clip(0, cap)
            net[o] = w * g["f"] - 0.001 * w.diff().abs().fillna(0)
        N = pd.DataFrame(net); N.index = N.index.astype(str)
        for p, (a, b) in P.items():
            X = N.loc[a:b].to_numpy().T; n = X.shape[1]
            st = rng.integers(0, n, (4000, int(np.ceil(n / 12)))); I = ((st[:, :, None] + np.arange(12)) % n).reshape(4000, -1)[:, :n]
            d = sr(X[2][I]) - sr(X[1][I]); lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"cap": cap or "none", "factor": fac, "period": p, "BH": sr(X[0]), "A": sr(X[1]), "B": sr(X[2]), "B-A": sr(X[2]) - sr(X[1]), "lo": lo, "hi": hi, "B-BH": sr(X[2]) - sr(X[0])})
T = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for cap in ("none", 2.0):
    print(f"\n=== cap {cap}")
    for p in P:
        print("--", p); print(T[(T.cap == cap) & (T.period == p)].drop(columns=["cap", "period"]).round(2).to_string(index=False))
