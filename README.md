# Volatility Timing: Forecasting and Decision-Focused Estimation

**Introduction to Financial Engineering, Hanyang University**  
Supervisor: Prof. Song Jae Wook  
Team: Ayush, Farres, Robin, Zheru

> **Status — 4 October 2026:** The revised research CSVs have passed internal consistency checks. Monthly factor returns and daily/monthly industry excess returns have also been reconciled with the corresponding Kenneth French source files. Next: implement the unmanaged, lagged-RV and GARCH(1,1) walk-forward benchmarks. No model-performance results are claimed yet. Parquet-to-CSV equality and the collection notebook still require verification.

## Contents

1. [Research direction](#1-research-direction)
2. [Experimental design](#2-experimental-design)
3. [Course and research layers](#3-course-and-research-layers)
4. [Data and definitions](#4-data-and-definitions)
5. [Loading and reproduction](#5-loading-and-reproduction)
6. [Forecasting and allocation](#6-forecasting-and-allocation)
7. [Evaluation](#7-evaluation)
8. [Implementation roadmap](#8-implementation-roadmap)
9. [Optional identification extension](#9-optional-identification-extension)
10. [References](#10-references)

## 1. Research direction

**Can GARCH-family and GARCH-informed neural forecasts improve volatility-managed portfolios relative to lagged realized variance, and does decision-focused training deliver greater out-of-sample economic value than statistical training after trading costs?**

The project proceeds in this order:

1. **Establish the baseline.** Use last month's realized variance as a backward-looking forecast for next month's risk, following the basic Moreira–Muir volatility-timing idea. Also retain unmanaged exposure as a benchmark.
2. **Compare forecasting methods.** Replace lagged RV with forecasts from GARCH(1,1), GJR-GARCH and a GARCH-informed neural network (GINN/GARCH-LSTM). Test both forecast accuracy and investment outcomes.
3. **Compare training objectives.** Within each fitted model, compare statistical estimation with decision-focused estimation aimed at portfolio utility after trading costs.
4. **Investigate identification only as an optional extension.** If the core results and time permit, study whether parameter uncertainty or weak identification helps explain forecast and allocation sensitivity.

Better variance forecasts need not produce better portfolios. Equally, better portfolio outcomes need not imply better variance forecasts. Measuring both, and explaining any disagreement, is central to the project.

Moreira and Muir (2017) motivate volatility management. Cederburg et al. (2020) motivate careful real-time evaluation: their findings emphasize that positive in-sample spanning alphas need not translate into implementable out-of-sample gains, with instability in the spanning relationships playing an important role. Their critique is not simply a claim that transaction costs remove all gains.

Decision-focused portfolio learning is an existing research area. We do not claim to invent it. The proposed contribution is a controlled comparison of forecasting methods and training objectives in volatility timing, including the mechanisms behind any economic gains or losses. Novelty remains subject to a focused literature review.

## 2. Experimental design

### 2.1 Models and objectives

| Method | Statistical estimation or training | Decision-focused estimation or training |
|---|---|---|
| Unmanaged exposure | Constant exposure benchmark | Not applicable |
| Lagged realized variance | Fixed backward-looking forecast; no fitted variance parameters | Not applicable to the fixed benchmark |
| GARCH(1,1) | Gaussian quasi-maximum likelihood (QMLE) | Same variance recursion, parameters selected using net portfolio utility |
| GJR-GARCH | Student-t likelihood with standardized, symmetric innovations | Same variance recursion and admissibility constraints, portfolio-based objective |
| GINN / GARCH-LSTM | Specified statistical variance loss and GARCH-informed structure or regularization | Same architecture, portfolio objective; a hybrid statistical/decision loss may be tested |

**GINN/GARCH-LSTM is part of the core project, not an optional extension.** Its size and training design must reflect the available data. The exact paper, architecture and meaning of “GARCH-informed” must be documented before implementation; a network that merely includes GARCH features should not be described as an exact replication of a different GINN architecture.

We make two distinct comparisons:

- **Across forecasting methods:** does each statistically trained model improve on lagged RV in forecast accuracy and portfolio performance?
- **Within a model:** does changing the objective improve economic outcomes, holding the architecture and implementation rules fixed?

Do not attribute a GARCH-versus-decision-focused-LSTM difference solely to the loss function: both model and objective changed.

### 2.2 What remains comparable

Use the same evaluation dates, assets, return definitions, monthly rebalancing schedule, allocation formula, leverage constraints and cost assumptions. Give methods a comparable tuning budget and the same access to historical information. Document any additional neural-network features.

Use a prespecified, historical-data-only rule for the scaling constant `c`. For the main within-model objective comparison, keep the calibrated constant fixed across the statistical and decision-focused versions. Any alternative recalibration policy must be applied consistently and reported separately.

Start with the market. Extend the frozen procedure to the other five factors and ten industries. Series share economic shocks: 254,352 series-day rows are not 254,352 independent observations.

## 3. Course and research layers

| | Course layer | Research layer |
|---|---|---|
| Purpose | Meet the course requirements | Test forecasting methods and estimation objectives |
| Data | Professor-provided CRSP monthly stock file, provisionally 2000–2020 | Kenneth French daily and official monthly data, July 1963–August 2026 |
| Intended output | Monthly rebalanced stock portfolio versus an appropriate S&P 500 total-return benchmark | Volatility-timed factor and industry portfolios |
| Current status | Return definitions, historical universe and assignment details still to verify | Revised CSVs checked; model implementation next |

Keep the datasets and their roles separate. The course application can later apply an approved market-level volatility overlay to a stock portfolio. That would not demonstrate accurate forecasts of each individual stock's volatility.

French portfolios and the course file may share underlying CRSP securities and dates. The course exercise is therefore not automatically independent validation. Genuine out-of-sample evidence comes from temporal holdouts and disciplined model selection, with held-out portfolio families providing an additional transfer check rather than statistical independence.

## 4. Data and definitions

### 4.1 Why this dataset

The research uses public factor and portfolio returns from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). They provide a long history and a reproducible connection to the volatility-timing literature. They are portfolio-level series, not an individual-stock CRSP database.

The earlier LSEG extraction based on a current stock screener is parked. A current-survivor universe can omit firms that exited before the screening date. A reliable historical universe is possible in principle, but it requires documented membership, identifiers, corporate actions and delisting treatment; casually appending dead-stock names does not solve the problem.

French portfolios avoid constructing our own current-survivor stock universe. This is not a blanket guarantee against every data bias. Historical revisions, formation rules and data vintage still matter. In particular, French changed its US source processing from CRSP FIZ to CIZ starting with the January 2025 release. Preserve the raw download vintage rather than assuming exact reproduction of older published results.

Daily estimation gives us more observations and captures within-month dynamics. We choose it for this design; monthly GARCH is not inherently invalid, and neither the size nor direction of finite-sample bias should be asserted universally.

### 4.2 Source files

All source files come from the French library. Preserve the raw ZIPs, retrieval dates and source headers.

| Frequency | Source ZIP | Purpose |
|---|---|---|
| Daily | `F-F_Research_Data_5_Factors_2x3_daily_CSV.zip` | Five factors and daily RF |
| Daily | `F-F_Momentum_Factor_daily_CSV.zip` | Momentum |
| Daily | `10_Industry_Portfolios_daily_CSV.zip` | Value-weighted industry returns |
| Monthly | `F-F_Research_Data_5_Factors_2x3_CSV.zip` | Official monthly factors and RF |
| Monthly | `F-F_Momentum_Factor_CSV.zip` | Official monthly momentum |
| Monthly | `10_Industry_Portfolios_CSV.zip` | Official value-weighted monthly industry returns |

The source files used for the latest reconciliation identify the **202608 CRSP database**. The collection notebook must be checked against this six-file workflow; it was not supplied for review.

### 4.3 Series and units

| Type | Series | Return definition |
|---|---|---|
| Market | `Mkt-RF` | Market excess return |
| Long–short factors | `SMB`, `HML`, `RMW`, `CMA`, `Mom` | Published factor spreads; do not subtract RF again |
| Industries | `NoDur`, `Durbl`, `Manuf`, `Enrgy`, `HiTec`, `Telcm`, `Shops`, `Hlth`, `Utils`, `Other` | Value-weighted industry total returns minus RF at the same frequency |

All stored returns use decimals: `0.01 = 1%`. Retain daily and official monthly RF separately. The current monthly table does not contain an RF column; add a keyed monthly RF series to the evaluation inputs before computing total-wealth metrics.

### 4.4 Coverage

| Measure | Current coverage |
|---|---|
| Daily span | 1 July 1963–31 August 2026 |
| Daily dates per series | 15,897 |
| Series | 16 |
| Long-format daily rows | 254,352 |
| Months per series | 758 |
| Monthly rows | 12,128 |

### 4.5 Dataset schema

Canonical repository names omit attachment suffixes such as `(1)`.

**`datasets/ff_daily_garchdata.parquet`** — daily model inputs.

| Column | Meaning |
|---|---|
| `Date` | Trading date |
| `series` | One of the 16 series |
| `ret` | Daily excess return or long–short factor return, in decimals |
| `month` | Calendar month, stored as `Period[M]` in the Parquet schema |

**`datasets/ff_monthly_baselinedata.parquet`** — monthly returns and variance targets.

| Column | Meaning |
|---|---|
| `series` | One of the 16 series |
| `month` | Calendar month |
| `ret_m` | Official monthly factor return, or official monthly industry total return minus official monthly RF |
| `rv` | Sum of squared daily `ret` observations in that month |
| `n_days` | Number of daily observations used |
| `rv_lag1` | Previous calendar month's RV for the same series |
| `incomplete` | Existing short-month flag, documented as `n_days < 15`; not a complete missing-date test |
| `month_end` | Calendar month-end label, not necessarily a trading date |

CSV counterparts are `ff_long.csv` and `monthly_returns_rv.csv`. `ff_merged.csv` contains wide daily data, including RF.

**Important correction:** `ret_m` is no longer calculated by compounding daily excess returns or daily factor spreads. Daily data supplies RV and model inputs; official monthly data supplies the monthly return used in the backtest.

### 4.6 Validation status

The following checks were performed on the revised CSVs on 4 October 2026:

| Check | Result |
|---|---|
| Wide-to-long daily reconciliation | Exact match |
| Duplicate daily/monthly keys | None found |
| Missing daily values or missing calendar months within series | None found in the supplied panel |
| Monthly RV reconstructed from daily squared returns | Matches within floating-point precision |
| `n_days` reconstructed from daily rows | Exact match |
| Within-series `rv_lag1` | Exact match; only the first month of each series is missing |
| Positive RV | All observations positive |
| Official monthly five-factor and momentum returns | All 758 months match |
| Official monthly industry returns minus monthly RF | All ten industries match |
| Official daily industry returns minus daily RF | All ten industries match |

Source-return differences were below `1e-16`. These checks do not constitute a full audit of the collection notebook, every original daily factor observation or the exchange calendar. Parquet schemas were inspected, but Parquet values were not independently reconciled with CSV values.

Before freezing the dataset:

- Verify Parquet-to-CSV equality after normalizing date and period types.
- Check expected trading dates, including historical special closures. September 2001 has 15 observations; a simple threshold cannot establish completeness.
- Detect French missing-value codes before converting percentages to decimals.
- Parse the intended table by its boundaries rather than relying on a fixed `nrows` count.
- Record value-weighted versus equal-weighted selection, source vintage and all join decisions. An inner join can conceal missing source dates unless checked beforehand.

## 5. Loading and reproduction

### Load model inputs

```python
import pandas as pd

daily = pd.read_parquet("datasets/ff_daily_garchdata.parquet")
monthly = pd.read_parquet("datasets/ff_monthly_baselinedata.parquet")

daily = daily.sort_values(["series", "Date"])
monthly = monthly.sort_values(["series", "month"])
mkt = daily.loc[daily["series"] == "Mkt-RF"].set_index("Date")["ret"]
```

Parquet requires a compatible engine such as `pyarrow`. If loading CSV instead, explicitly parse `Date`, `month_end` and `month`. Normalize attachment filenames to the canonical repository names before using these paths.

### Required collection workflow

Update or verify `Data_Collection.ipynb` so that it:

1. Reads the six daily/monthly sources and preserves raw files and download metadata.
2. Selects the correct tables, detects missing-value codes and converts percent returns to decimals.
3. Checks source date coverage before merging daily data.
4. Subtracts daily RF from daily industry total returns, and official monthly RF from official monthly industry total returns. It leaves published factor returns unchanged.
5. Computes each month's `rv = sum(ret ** 2)` and `n_days` from daily observations.
6. Joins those statistics to official monthly returns on unique `(series, month)` keys.
7. Checks calendar continuity, then constructs `rv_lag1` within each series.
8. Saves daily and monthly RF for evaluation, runs validation, and verifies CSV/Parquet equivalence.

This is the required workflow, not a claim that the unreviewed notebook already implements every step.

## 6. Forecasting and allocation

### 6.1 Timing and target

Let `t` denote the completed calendar month. At its end, forecast next month's variance using only information available through `t`. Apply the resulting exposure during `t+1`.

$$
RV_t=\sum_{d\in t}r_d^2,
\qquad
\widehat V^{RV}_{t+1\mid t}=RV_t.
$$

Thus, in a dataset row labelled month `t+1`, `rv_lag1` is the baseline forecast and `rv` is the realized target. Never use that row's contemporaneous `rv` to choose its exposure.

Our RV is an uncentered sum of squared daily returns, not demeaned daily sample variance. Describe this as a Moreira–Muir-style baseline and check the original paper's exact normalization before claiming an exact replication.

### 6.2 GARCH and GJR forecasts

For GARCH(1,1), with daily innovation `epsilon_d`:

$$
h_{d+1}=\omega+\alpha\epsilon_d^2+\beta h_d.
$$

Under stationarity, define `p = alpha + beta` and `h_bar = omega / (1 - p)`. If `D` is the final observed trading day, the forecast sum of next month's daily conditional variances is

$$
\widehat V_{t+1\mid t}
=\sum_{j=1}^{H}E_D[h_{D+j}]
=H\bar h+(h_{D+1\mid D}-\bar h)\frac{1-p^H}{1-p}.
$$

Use a numerically stable recursion near `p = 1` and compare the calculation with `arch` multi-step forecasts. Use the number of scheduled sessions known at the forecast origin for `H`; document an approximation such as 21 if used. Do not retrospectively reveal unscheduled closures to the forecast.

This is a sum of expected daily innovation variances. With zero conditional daily means it also forecasts our squared-return RV target. If nonzero means are modeled, include the corresponding expected squared-mean terms. It is not an exact formula for the variance of a compounded monthly simple return.

GJR-GARCH adds `delta * epsilon_d**2 * I(epsilon_d < 0)`. Under standardized symmetric innovations with finite variance, expected persistence is `alpha + beta + delta / 2`. Enforce positivity, the chosen stationarity constraint and, for Student-t innovations, degrees of freedom above two.

If fitting returns in percentage units (`ret * 100`), convert predicted variances back by dividing by `100**2`.

### 6.3 GINN / GARCH-LSTM

Implement a compact GARCH-informed neural model after validating the classical pipeline. Specify whether GARCH enters the architecture, inputs, loss or a combination. A direct positive next-month variance output is a practical candidate. A recursively simulated daily architecture is another option, but simulation is not automatically required merely because the model contains an LSTM.

Keep architecture, inputs and training budget comparable across statistical and decision-focused versions. Fit feature scalers and any GARCH-derived inputs using historical data available at each relevant origin. GARCH features generated from a full-sample fit would leak future information.

The dataset offers 15,897 daily observations but only 758 non-overlapping monthly targets per series. Control model size, use chronological validation and report multiple training seeds. Do not randomly split rows across dates or allow overlapping target windows to cross a holdout boundary.

### 6.4 Allocation, costs and training objective

The implementable comparison uses a common inverse-variance rule:

$$
w_t=\operatorname{clip}\left(\frac{c}{\max(\widehat V_{t+1\mid t},v_{\min})},0,w_{\max}\right).
$$

Choose `c`, the variance floor and the exposure cap using development data and keep the policy fixed for the reported test. A full-sample-normalized replication may be shown separately as descriptive evidence, not as an implementable strategy. Positive constant scaling alone leaves a frictionless Sharpe ratio unchanged; leverage, caps, costs and utility make scaling economically consequential.

Write net monthly excess performance as

$$
r^{net}_{t+1}=w_t r^e_{t+1}-\kappa\tau_t-b_t,
$$

where `tau_t` is turnover under an explicitly defined implementation, `kappa` is its unit cost, and `b_t` represents any additional financing or borrowing cost. For long-only portfolios, turnover should account for the pre-trade weight after market drift. For long–short factors, exposure changes alone do not capture constituent turnover, shorting costs or all trading expenses. Label a simplified exposure-change cost calculation as a proxy and test cost sensitivity.

The decision-focused objective is the sample mean–variance criterion:

$$
\widehat J(\theta)
=\overline{r^{net}(\theta)}
-\frac{\gamma}{2}\widehat{\operatorname{Var}}[r^{net}(\theta)].
$$

Optimize it on historical training paths, select hyperparameters using chronological validation, and evaluate the resulting procedure on later dates. Historical future returns can serve as training labels; they cannot enter a forecast before their availability date or influence choices using the held-out test period.

A hybrid loss may combine this objective with a statistical variance loss, with scaling and mixture weight selected using development data. Do not automatically call a decision-optimized output a calibrated variance forecast. Evaluate calibration and forecast loss separately.

**Exposure control:** a decision-focused model could improve utility by shifting forecast levels and hence average leverage. Report exposure and realized risk, and include a simple historically calibrated exposure-scaled statistical benchmark. This distinguishes timing improvements from changes in risk-taking.

## 7. Evaluation

### 7.1 Chronological protocol

Choose the initial estimation window, validation period and final test period before comparing model results. A possible initial fitting endpoint is December 1999; the complete split remains to be agreed. Do not present an illustrative split as an already executed experiment.

At each monthly forecast origin, fit or update using only available history, forecast the next month, record the allocation, and then evaluate its realized outcome. Use the same origins across models. Once a test month ends, it may enter a scheduled expanding-window update, but test performance must not be used to redesign the method.

Specify whether parameters are refitted monthly or less frequently; filtering the volatility state is distinct from re-estimating parameters. Keep hyperparameter selection inside the historical development process. All series share calendar cutoffs.

### 7.2 Metrics

| Question | Measures |
|---|---|
| Are variance forecasts better? | QLIKE and MSE against monthly RV; forecast bias and calibration |
| Are portfolios economically better? | Net Sharpe ratio, annualized return, maximum drawdown and mean–variance utility |
| What changes in implementation? | Turnover, cost drag, average/maximum exposure, realized volatility and cap frequency |
| Where do gains or losses arise? | Subperiods, factor/industry groups, and prespecified post-spike exposure analysis |

One dimensionless QLIKE form is `RV / V_hat - log(RV / V_hat) - 1`, for positive target and forecast. Use the same months and target convention for every method.

Calculate Sharpe from monthly net excess returns and state the annualization convention. Calculate compounded annual growth and drawdown from a defined total-return wealth series, including monthly RF and the financing/collateral convention. Do not silently compound excess-return spreads as if they were standalone total returns. A sample mean–variance score is a utility approximation, not automatically an exact certainty equivalent for every preference specification.

Report uncertainty in paired performance differences, using an approach that respects time dependence and common shocks across series. Keep the number of model variants manageable and disclose all tested variants rather than selecting only winners.

Define volatility spikes using a historical-information rule before examining test outcomes. Exposure recovery is a mechanism diagnostic, not proof that faster re-entry is always desirable.

## 8. Implementation roadmap

| Stage | Deliverable | Completion condition |
|---|---|---|
| 1. Freeze data and protocol | Reproducible collection, monthly RF, agreed splits and conventions | Validation documented; pending checks resolved |
| 2. Establish market benchmarks | Unmanaged, lagged-RV timing and GARCH(1,1) | Correct forecast timing, monthly aggregation and wealth accounting |
| 3. Expand statistical forecasts | GJR-GARCH and GINN/GARCH-LSTM | Comparable forecast and strategy results on common dates |
| 4. Add decision-focused training | Within-model objective comparisons | Same architecture/rules; chronological tuning; exposure controls |
| 5. Test robustness and explain results | Remaining factors, industries, costs and post-spike diagnostics | No retuning on held-out families; uncertainty reported |
| 6. Complete course application | Approved CRSP stock-portfolio exercise | Verified stock-return definitions and historical universe |
| 7. Optional research extension | Parameter uncertainty/identification analysis | Core project complete; scope agreed with supervisor |

### Open tasks

- Verify the revised collection notebook and CSV/Parquet equivalence.
- Save official monthly RF; distinguish calendar labels from actual forecast-origin trading dates.
- Pin the GINN reference and architecture, model windows, loss definitions and reproducible seeds.
- Agree chronological splits, exposure caps, scaling, cost conventions and evaluation metrics before model selection.
- Verify whether the professor's `ADJ_PRC` includes dividends; prefer documented total-return fields when available. There is no universal fixed percentage correction for omitted dividends.
- Verify the CRSP extract's historical inclusion rules, permanent identifiers, exits and delisting-return treatment. A declining stock count does not prove survivorship safety, and a flat count does not prove screening.
- Keep the current-screener LSEG pull off the critical path; Farres was the previously assigned owner for pausing it. Confirm its operational status within the team.

## 9. Optional identification extension

After the core experiment, ask whether parameter combinations with similar statistical support imply different persistence, monthly forecasts or speeds of restoring exposure after spikes.

Start with parameter sensitivity and estimation uncertainty. Formal weak-identification claims require model-specific justification; an unstable estimate, flat objective or large standard error alone is not sufficient proof. Stable forecasts despite uncertain parameters would also be informative.

QMLE is a practical initial estimator, not a restriction on later research. GMM or IV may be considered if appropriate moment conditions, valid and relevant instruments, and a clear identification question are established with the supervisor. They are not automatically required by conditional moment restrictions, and they do not automatically solve weak identification.

This extension must not delay the forecasting and decision-focused comparisons.

## 10. References

- Moreira, A. and Muir, T. (2017). *Volatility-Managed Portfolios*. Journal of Finance, 72(4), 1611–1644. [DOI](https://doi.org/10.1111/jofi.12513).
- Cederburg, S., O'Doherty, M. S., Wang, F. and Yan, X. (2020). *On the Performance of Volatility-Managed Portfolios*. Journal of Financial Economics, 138(1), 95–117. [DOI](https://doi.org/10.1016/j.jfineco.2020.04.015).
- Wang, F. and Yan, X. (2021). *Downside Risk and the Performance of Volatility-Managed Portfolios*. Journal of Banking & Finance, 131. [Publisher](https://www.sciencedirect.com/science/article/pii/S0378426621001576).
- Fama, E. F. and French, K. R. (2015). *A Five-Factor Asset Pricing Model*. Journal of Financial Economics, 116(1), 1–22.
- Glosten, L., Jagannathan, R. and Runkle, D. (1993). *On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks*. Journal of Finance, 48(5), 1779–1801.
- [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html): source data, construction notes and historical-vintage information.
- [`arch` forecasting documentation](https://arch.readthedocs.io/en/latest/univariate/forecasting.html): analytic, simulation and bootstrap forecast conventions.

**Literature tasks before claiming novelty:** add the exact selected GINN paper and architecture; review decision-focused portfolio-learning studies and distinguish their settings from this experiment. Do not treat an unverified author/year citation as an established reference.
