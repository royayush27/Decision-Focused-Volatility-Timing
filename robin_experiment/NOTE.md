# Robin's Experiment 1 (decision-trained volatility models), copied into the final version

Author: Robin. Original location: `Robin/experiment1_vol_managed/volexp/` in the project folder. Report: `experiment1_report.pdf`
(was `Robin/Untitled.pdf`). His own README explains every file.

- Code and his result tables are unchanged. The pickled panels (`results/bt_*.pkl`, `results/zhao/`, `results/seeds/`) are not
  copied; `backtest.py` and `backtest_zhao.py` re-create them (about 1.5 h on 2 cores), and `convert_pickles.py` turns them into
  `../data/robin/<factor>_forecasts.csv`, which `../9_dfl_robin.py` reads.
- `../9_dfl_robin.py` first reproduces his reported Sharpe ratios from those CSVs (largest difference 0.0005), then evaluates his
  forecasts under our frozen protocol (SPEC.md).
- His protocol differs from ours in three ways: no leverage cap, sample 1937 (1974 for RMW, CMA) to Aug 2026, c set on
  in-sample fitted forecasts. Our script 9 re-applies our c rule and cap to his forecasts.
