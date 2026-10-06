"""The frozen protocol (see SPEC.md). Every script reads its settings from here, so no script can disagree with another.

Run any script with  --spec main      (default, the protocol in SPEC.md)
                     --spec old       (our earlier settings: 5 bp, c frozen on pre-1998 data, cap 2)
                     --spec uncapped  (main, but no leverage cap: the Moreira-Muir / Robin setting)
Results go to results/<spec>/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "results" / "cache"          # GARCH forecasts (identical for every spec)

SPECS = {
    "main": dict(cost=0.0010,            # 10 bp per unit of exposure change (Moreira-Muir Table IV)
                 cap=1.5,                # exposure w in [0, 1.5] (Moreira-Muir Table V)
                 c_rule="rematch",       # c re-set every January on all past months: managed vol = unmanaged vol
                 min_cal_months=24),     # months of past forecasts needed before the first c
    "old": dict(cost=0.0005, cap=2.0, c_rule="frozen", min_cal_months=24),   # c: mean exposure 1 on the pre-1998 window
    "uncapped": dict(cost=0.0010, cap=None, c_rule="rematch", min_cal_months=24),
}

PERIODS = {"1998-2025": ("1998-01", "2025-12"),      # main horizon
           "1998-2015": ("1998-01", "2015-12"),
           "2016-2025": ("2016-01", "2025-12"),      # last 10 years
           "2021-2025": ("2021-01", "2025-12")}      # last 5 years
MAIN = "1998-2025"

BLOCK = 12            # months per bootstrap block (circular block bootstrap)
N_BOOT = 5000         # bootstrap draws
SEED = 0
CEDERBURG_MIN = 120   # months of past managed returns before the real-time combination starts

US_FIRST_GARCH_YEAR = 1970      # US data start in 1963
INTL_FIRST_GARCH_YEAR = 1990    # earliest refit abroad (later where data start later)


def spec_name():
    a = sys.argv
    return a[a.index("--spec") + 1] if "--spec" in a else "main"


def spec():
    s = dict(SPECS[spec_name()]); s["name"] = spec_name(); return s


def out_dir():
    d = ROOT / "results" / spec_name(); d.mkdir(parents=True, exist_ok=True); return d
