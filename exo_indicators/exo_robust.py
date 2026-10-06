"""Robustness + event study for the exogenous-indicator test (imports exo_test)."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd

import exo_test as E

ind, fc, X, y, B = E.ind, E.fc, E.X, E.y, E.B
L1 = ind.shift(1).reindex(fc.index)
emv_rel = ind["EMV"] / ind["EMV"].expanding().median()           # attention relative to its own past median
# looser timing (value of t-1 even where publication lags by days) and more timely credit spread
X["lEMV_L1"] = np.log(L1["EMV"])
X["NS_L1"] = L1["NS_avg"]
X["DEFe"] = L1["DEF_eom"]                                         # Baa-Aaa at the last trading day of t-1
X["dDEFe"] = L1["DEF_eom"] - ind["DEF_eom"].shift(2).reindex(fc.index)
# attention-weighted negative sentiment: bad tone counts more when the press is focused on market volatility
X["AWS"] = (-ind["NS_avg"] * emv_rel).shift(2).reindex(fc.index)
X["AWS_L1"] = (-ind["NS_avg"] * emv_rel).shift(1).reindex(fc.index)

E.evaluate("1986-03", "1995-01", ("2005-01", "2024-12"), ("1995-01", "2004-12"),
           ["DEFe", "dDEFe", "lEMV_L1", "NS_L1", "AWS", "AWS_L1"], "Robustness (looser timing, eom spread, attention-weighted tone)")

# event study: biggest GARCH surprises 1990-2025, indicator z-scores as known k months before
sur = (y - B["lg"]).loc["1990-01":"2025-12"]
ev = sur.sort_values(ascending=False).index[:12]
cols = {"dDEFe": X["dDEFe"], "CS": X["CS"], "lEPU": X["lEPU"], "lEMV_L1": X["lEMV_L1"], "-NS_L1": -X["NS_L1"],
        "AWS_L1": X["AWS_L1"], "lVIX2": X["lVIX2"], "lagRV": B["lr"]}
pos = {m: i for i, m in enumerate(fc.index)}
print("\nTop-12 GARCH-surprise months:", ", ".join(ev))
print("z-score (vs prior 10 years) of each indicator as known at the forecast date, k months before the spike month")
tab = {}
for name, s in cols.items():
    z = (s - s.rolling(120, min_periods=60).mean()) / s.rolling(120, min_periods=60).std()
    tab[name] = {k: np.nanmean([z.iloc[pos[m] - k] for m in ev]) for k in (3, 2, 1, 0)}
T = pd.DataFrame(tab).T
T.columns = [f"k={k}" + (" (forecast date)" if k == 0 else "") for k in T.columns]
print(T.round(2).to_string())
print("\nk=0 is the information set used to forecast the spike month; k=1..3 are earlier month-ends.")
