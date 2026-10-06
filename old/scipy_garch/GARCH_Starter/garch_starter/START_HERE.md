# GARCH(1,1): first forecasting implementation

This is the first code milestone for the volatility-timing project. It compares
GARCH forecasts with lagged realized variance. It does **not** yet backtest an
allocation rule, calculate Sharpe ratios or claim that GARCH is more profitable.

## Run

Extract the ZIP and open a terminal in `garch_starter`:

```bash
python -m pip install -r requirements.txt
jupyter lab 01_GARCH_Market.ipynb
```

Or run the same implementation without Jupyter:

```bash
python garch_baseline.py
python -m unittest -v test_garch_baseline
```

The notebook imports `garch_baseline.py`, so keep them in the same folder.
The ZIP includes market-only extracts of your supplied CSVs; no data download is
needed. To use the full original long files, pass `--daily PATH --monthly PATH`.
The loader also accepts Parquet files. Decimal-return input is required.

## Choices made before this development run

- Daily `Mkt-RF`, not raw market prices and not compounded daily excess returns.
- Zero conditional mean, Gaussian QMLE, GARCH(1,1); robust covariance.
- Expanding history starting July 1963; refit at each month-end.
- First fit ends December 1999. Development targets are January 2000–December
  2009 (120 forecasts). Data after 2009 are not used for this performance run.
  This development sample is not an untouched final test. Agree validation and
  final holdout dates before comparing/tuning the later models.
- Forecast the sum of the next **21** daily conditional variances. This fixed
  horizon is known at the origin and avoids using a realized future day count.
  Calendar months vary in length, so it is an approximation to their actual RV.
  A scheduled-session horizon is a later refinement, not an optimized choice.
- Lagged RV means the preceding calendar month's sum of squared daily excess
  returns. The GARCH target is next month's same uncentered RV under zero mean.
- Fit returns in percentage units, then divide all variance forecasts and omega
  by 10,000 before export. Alpha and beta do not change units.
- No silent dropping of failed fits. The code retries once, records warnings and
  convergence information, then raises if a valid fit cannot be obtained.
- Multi-step forecasts are cross-checked against an independent expectation
  recursion. Finite-horizon forecasting does not divide by `1-alpha-beta` near
  the integrated boundary. An unconditional variance is not reported there.

## Read the notebook in order

1. Understand the model and timing.
2. Inspect the initial fit through 1999.
3. Compute the January 2000 forecast and compare the daily recursion.
4. Run the expanding-window development experiment.
5. Compare QLIKE and MSE, then inspect forecasts and persistence.

### Executed development results

All 120 monthly fits converged without retries or recorded warnings. Four focused
checks passed: future-return perturbation, independent aggregation/closed-form
agreement, month alignment/missing-target handling, and zero loss for perfect
forecasts. The initial 1963–1999 fit estimates alpha at about 0.08393 and beta at
0.91033 (persistence 0.99425).

| Model | QLIKE | MSE (decimal variance squared) |
|---|---:|---:|
| Lagged RV | 0.25241251 | 0.000024660473 |
| GARCH(1,1) | 0.21377754 | 0.000019095148 |

GARCH's development losses are lower by 15.31% and 22.57%, respectively. Its mean
forecast is about 94.77% of mean realized RV on these months. No statistical
significance or superiority on a final holdout is claimed.

Lower QLIKE and MSE indicate better forecasts for the specified proxy, not better
portfolio performance. The next milestone is the unmanaged/lagged-RV/GARCH
allocation comparison with historical-only scaling, monthly RF, costs and a
defined total-return wealth calculation.

## Files

| File | Role |
|---|---|
| `01_GARCH_Market.ipynb` | Teaching notebook with executed outputs |
| `garch_baseline.py` | Reusable estimator, walk-forward loop and scoring functions |
| `test_garch_baseline.py` | Leakage, timing, aggregation and loss checks |
| `datasets/market_daily.csv` | Market daily extract from the revised long CSV |
| `datasets/market_monthly.csv` | Market monthly extract; official monthly returns retained |
| `results/market_forecasts.csv` | Forecasts, outcomes, parameters and diagnostics by month |
| `results/forecast_metrics.csv` | Development forecast metrics |
| `results/run_manifest.json` | Configuration, core versions and input hashes |
| `results/forecast_comparison.png` | Realized/forecast volatility and persistence plot |
| `results/initial_fit.txt` | Initial fit summary through December 1999 |
| `results/environment_versions.json` | Package versions used for this run |

The two supplied Parquet datasets were compared to their revised CSV counterparts
on 4 October 2026 after installing a compatible reader. Values match to numerical
precision after normalizing date/period types. The collection notebook itself
was not provided and has not been audited.

The Gaussian objective is used as QMLE; it does not establish that innovations
are normally distributed. This first implementation makes no weak-identification
claims and does not select the best GARCH variant. GJR and GINN remain subsequent
core model comparisons.

Sources: [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
and [`arch` forecasting documentation](https://arch.readthedocs.io/en/latest/univariate/forecasting.html).
Input extracts preserve the user's supplied research data and its 202608 source
vintage. Source-code and dataset licensing are distinct; no blanket software
license for upstream data is implied.
