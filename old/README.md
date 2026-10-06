# old/  (archive)

Nothing here is needed for the current work. Kept so every earlier number can be traced back.

| Folder | What it is |
|---|---|
| `scipy_garch/` | The scipy / `arch` GARCH pipeline. `garch_baseline.py` produced `results_1970/market_forecasts.csv` and `stat_forecasts.py` produced `results_1970/forecasts/*_stat.csv`, which the backtests still read. The torch GJR in `ginn/GINN_v2.ipynb` is now the main GARCH code. Also `garch_x.py` / `rules.py` / `plot_vix.py` (GARCH-X with VIX and VIX rules, outputs in `forecasts_vix/` and `figures/`), `evaluate.py`, `compare_dist.py`, the starter kit and notebooks. |
| `scipy_garch/results_first_run/` | The first baseline run (the old `results/`). `results_first_run_copy/` is an identical duplicate of it and can be deleted. |
| `ginn_v1/` | First GINN notebook (2,500 days) and its two backups. |
| `ginn_v2_run1/` | GINN v2, first run, with its printed outputs. GJR-X had not converged in this run (loss 0.38835 vs 0.38721), so use `ginn/GINN_v2.ipynb` instead. |
| `misc_stuff/` | Early data scripts and the progress-update slides. |

The archived scripts still run from the project root (they change into it), e.g. `python old/scipy_garch/garch_baseline.py`.
Careful: with its defaults that script rewrites `results_1970/`.
