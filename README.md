# Exploratory work (Steps 0-7 of CHRONOLOGY.md)

This branch is a snapshot of our working folder: every script we ran **before** the protocol was frozen. The final, citable
results are on `main` (the final-version code). Numbers here use our earlier settings (5 bp costs, c frozen so mean exposure
= 1 on 1970-1999, cap 2; Steps 1-5 evaluate 2000-2024), so they differ slightly from `main`.

Paths match `CHRONOLOGY.md` on `main`: a path written `fe_project/X` there is `X` on this branch.

| Folder / file | CHRONOLOGY step | What it is |
|---|---|---|
| `data/` | 0 | `Data_Collection.ipynb` (download and checks) and the Ken French, JKP and regional data files |
| `old/scipy_garch/`, `calibrate_c.py`, `backtest.py`, `results_1970/` | 1 | First GARCH / GJR forecasts, c calibration, backtest and accuracy tests |
| `dfl_diagnosis/` | 2 | Why decision-focused training does not help: MT x VT, decision-loss GARCH, Monte Carlo of the theory, cost-aware losses |
| `exo_indicators/` (README items 1-12) | 3 | Better information: spreads, uncertainty and news indices, GARCH-X |
| `ginn/` | 4 | GARCH-informed neural network (GINN), torch GJR, ensembles |
| `exo_indicators/` (README items 13-20) | 5 | Fixing the return side: VIX^2, VRP, Xu switches, ICR, rebalancing speed, VIX term structure, recovery state, Kalman filters |
| `main_eval/` | 6-8 (first versions) | 1998-2025 market, panel, factor mixes, international (JKP, French) with the earlier settings |
| `old/ginn_v1`, `old/ginn_v2_run1`, `old/misc_stuff` | | Superseded first attempts, kept for the record |
| `3-Analysis/`, `1-References/` | | Literature matrix and references |
| `WORKING_README.md` | | The working README of the folder at that time |

Every signal tested here was insignificant on its own; see `CHRONOLOGY.md` on `main` for the reasoning at each step.
