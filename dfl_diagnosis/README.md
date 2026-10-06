# dfl_diagnosis

Scripts behind the "why decision-focused training does not help volatility timing" analysis.
Run every command from the project root (the scripts change into it anyway). Inputs used:
`data/ff_daily_garchdata_long.csv`, `data/ff_monthly_baselinedata.csv`, `data/ff_monthly_rf.csv`,
`results_1970/market_forecasts.csv`, `results_1970/forecasts/*.csv`, plus `backtest.py`, `calibrate_c.py`, `dfl_diagnosis/decision_garch.py`.

Run order (times on a laptop-class CPU):

1. `python dfl_diagnosis/decomp.py`
   Market only. Splits each strategy's Sharpe ratio into mean timing (MT) and variance timing (VT),
   SR_managed / SR_unmanaged = MT x VT, plus VR_rv (managed / unmanaged variance using realised variance).
2. `python dfl_diagnosis/risk_dfl.py --series Mkt-RF SMB HML RMW CMA Mom NoDur Durbl Manuf Enrgy HiTec Telcm Shops Hlth Utils Other`
   About 30 s per series. GARCH and GJR trained three ways (qmle, util, risk), annual refits 2000-2025.
   Writes `dfl_diagnosis/panel/<series>_<model>.csv`.
3. `python dfl_diagnosis/panel_eval.py`
   Panel table and paired block-bootstrap tests (6-month blocks shared across series).
4. `python dfl_diagnosis/mc_theory.py 40`
   Monte Carlo for the theory (cases A, B, C). Use 200 reps for the final write-up (about 1 hour).
5. `python dfl_diagnosis/cost_dfl.py 0.0025` and `python dfl_diagnosis/cost_dfl.py 0.0050`
   Cost-aware risk loss. Then `cost_eval.py` (vs qmle, unmanaged) and `cost_partial.py` (vs partial adjustment).
6. `python dfl_diagnosis/pooled_curvature.py`
   Learn the exposure curvature p in w = c * v_hat^(-p), per series vs pooled across 16 series.

Losses:
- qmle: Gaussian quasi-likelihood on daily returns.
- util: realised net utility of the managed strategy on training months (uses returns).
- risk: mean(rv * w^2) with mean(w) = 1 on training months (uses realised variance, no returns).
Decision losses keep omega at its QMLE value (utility is scale invariant); every forecast is re-anchored
so its training-window mean equals mean realised variance, and all three use the same calibration window.
