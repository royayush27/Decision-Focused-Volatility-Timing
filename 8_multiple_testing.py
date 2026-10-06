"""8. Multiple testing: which timing gains survive once we account for how many things we tried?

Family = every 'timed vs unmanaged' comparison saved by scripts 1-7 and 9 (results/<spec>/nets_*.csv) on the main horizon 1998-2025,
for series that have all 336 months. Duplicates (identical return series, e.g. 'legs' of a one-factor mix) and comparisons
where the timed series equals the unmanaged one are dropped. Each test: Sharpe gain, bootstrap standard error, t, two-sided p.
ONE joint circular block bootstrap (same resampled months for every test, so the correlation between tests is kept),
then Holm (family-wise), Benjamini-Hochberg (false discovery rate) and Romano-Wolf step-down (family-wise, uses the joint draws).
Not in the family: the exploratory signals in the old project folders (VIX/VRP, ICR, Kalman, GARCH-X, GINN...), which ran on
other periods. All of them were insignificant on their own; adding them would only make the corrections stricter.
Also not a timing test: the hindsight choice of Mkt + RMW + CMA + Mom among 63 mixes (see script 4 for its percentile).
"""
import glob

import numpy as np
import pandas as pd

import config
from core import report
from core.stats import sharpe, boot_idx, sr_rows, holm, bh, romano_wolf
from scipy.stats import norm

S = report.start("8_multiple_testing", "8. Multiple testing across every timing comparison (1998-2025)")
a, b = config.PERIODS[config.MAIN]
cols = {}
for f in sorted(glob.glob(str(config.out_dir() / "nets_*.csv"))):
    X = pd.read_csv(f, index_col="month").loc[a:b]
    src = f.split("nets_")[-1][:-4]
    for c in X.columns:
        cols[(src, c)] = X[c]
n = len(pd.RangeIndex(0, 336))
tests, seen = [], set()
for (src, c), z in cols.items():
    g, s, rule = [x.strip() for x in c.split("|")]
    if rule == "Unmanaged":
        continue
    u = cols.get((src, f"{g} | {s} | Unmanaged"))
    if u is None or z.notna().sum() != n or u.notna().sum() != n:
        continue
    key = tuple(np.round(z.to_numpy(), 10))
    if np.allclose(z.to_numpy(), u.to_numpy()) or key in seen:
        continue
    seen.add(key); tests.append((src, g, s, rule, z.to_numpy(), u.to_numpy()))
I = boot_idx(n)
rows, draws = [], []
for src, g, s, rule, z, u in tests:
    d = sharpe(z) - sharpe(u); db = sr_rows(z[I]) - sr_rows(u[I]); se = db.std(ddof=1)
    rows.append({"script": src, "group": g, "series": s, "rule": rule, "dSR": d, "se": se, "t": d / se,
                 "p_raw": 2 * (1 - norm.cdf(abs(d / se)))})
    draws.append((db - d) / se)
T = pd.DataFrame(rows)
T["p_holm"], T["p_bh"] = holm(T.p_raw), bh(T.p_raw)
T["p_romano_wolf"] = romano_wolf(T.t, np.vstack(draws))
T = T.sort_values("p_raw").reset_index(drop=True)
T.to_csv(config.out_dir() / "8_multiple_testing.csv", index=False)
m = len(T)
print(f"{m} timing comparisons in the family, {n} months, one joint bootstrap ({config.N_BOOT} draws, {config.BLOCK}-month blocks)\n")
print(f"significant at 5%: raw {int((T.p_raw < .05).sum())} (about {0.05 * m:.0f} expected by luck alone), "
      f"Benjamini-Hochberg {int((T.p_bh < .05).sum())}, Holm {int((T.p_holm < .05).sum())}, Romano-Wolf {int((T.p_romano_wolf < .05).sum())}\n")
print("The 25 smallest raw p-values")
report.show(T.head(25).drop(columns=["script"]), 3)
HEAD = [("US market", "Mkt-RF", "MM on GARCH"), ("US mix", "Mkt + RMW + CMA + Mom", "MM on GARCH"),
        ("US mix", "RMW + CMA + Mom", "MM on GARCH"), ("US series", "Mom", "MM on GARCH"),
        ("JKP world_ex_us", "Mkt + Prof + Inv + Mom", "MM on GARCH"), ("JKP world_ex_us", "Mom alone", "MM on GARCH"),
        ("French Developed_ex_US", "Mkt + RMW + CMA + Mom", "Whole mix"), ("French Developed_ex_US", "RMW + CMA + Mom", "Whole mix"),
        ("French Developed_ex_US", "Mom alone", "Whole mix"),
        ("Robin DFL", "RMW", "ZHAO_B"), ("Robin DFL", "Mom", "ZHAO_B")]
H = pd.concat([T[(T.group == g) & (T.series == s) & (T.rule == r)] for g, s, r in HEAD])
print("Headline comparisons, adjusted within the full family"); report.show(H.drop(columns=["script"]), 3)
