# Research chronology: how we got from the question to the final results

Read this before the meeting. For every step it gives the question, the reasoning [in brackets], the code and data files,
how to run them, what came out, and whether the step is in `final-version/` or only in the working project folder.

**Two kinds of numbers.**
- **Final numbers** come from `final-version/` (scripts 1-9) under the frozen protocol in `SPEC.md`: 10 bp costs, c re-set every
  January on past data only, exposure cap 1.5, 12-month bootstrap blocks, Cederburg test, Romano-Wolf. These are the ones to
  present as results.
- **Exploratory numbers** (Steps 1-5) come from scripts in the working folder `fe_project/` (not in final-version) with our
  earlier settings: 5 bp, c frozen on a calibration window, cap 2, 6-month blocks, evaluation 2000-2024. They explain how we
  reached the design and the thesis; present them as exploratory evidence, not as headline results.

Paths below starting with `fe_project/` are in the working folder, which is on the GitHub branch `exploratory` (drop the
`fe_project/` prefix there); paths without a prefix are inside `final-version/`, which is `main` on GitHub.
Everything runs from the folder root (`python <script>`). Requirements: numpy, pandas, scipy (plus torch and arch for Robin's
code and the GINN notebook).

---

## Step 0. Question and data (September)

**Question.** Volatility timing (Moreira and Muir 2017): exposure w = c / forecast variance, rebalanced monthly. Cederburg et al.
(2020) showed most of the gain disappears in real time. Every paper fits the variance model for accuracy (maximum likelihood)
and then uses it for a portfolio decision. Does fitting it for the decision instead (decision-focused learning, DFL) do better?

[Why Ken French data and not the 3,300-stock LSEG pull: the LSEG list came from today's screener, so dead firms were missing
(survivorship bias), and failed firms are exactly the extreme-volatility events. French portfolios are built from CRSP with
delisted firms, are the literature standard (MM 2017 and Cederburg 2020 use them), and are free to replicate.]
[Why daily data: GARCH fitted on ~250 monthly points is badly biased toward low persistence, which flattens the forecast and
makes GARCH look worse than it is. Rule: fit on daily returns, aggregate to a monthly forecast.]
[Why excess returns: factors already are; the 10 industries are total returns, so RF is subtracted.]

| File | What it contains |
|---|---|
| `fe_project/data/Data_Collection.ipynb` | Downloads the three French zips, reads them, merges, checks, writes the files below |
| `data/ff_daily_garchdata_long.csv` | Daily returns, long format: Date, series (16 names), ret (decimal), month. 1963-07 to 2026-08, 15,897 days x 16 |
| `data/ff_monthly_baselinedata.csv` | Monthly: series, month, ret_m (compounded), rv (sum of squared daily returns), n_days, rv_lag1, incomplete, month_end |
| `data/ff_monthly_rf.csv` | Monthly T-bill rate RF (decimal), 1926-07 on (1926-1963 added from Robin's French file; identical where they overlap) |
| `data/VIXCLS.csv` | Daily VIX from FRED, 1990 on |

Checks that pass: worst Mkt-RF day = 19 Oct 1987 (-17.4%); annualised vol from RV = from daily sd (16.21%).
**Verify:** open the CSVs; `python 1_us_market.py` reads them.

---

## Step 1. First forecasts and the Moreira-Muir baseline (exploratory, 2000-2024)

**Question.** Does a GARCH forecast beat last month's realised variance (the MM rule) for timing the market?
[c set on 1970-1999 only (training data), never on the full sample: the full-sample c is one of Cederburg's look-ahead critiques.]

| File | What it does / contains |
|---|---|
| `fe_project/old/scipy_garch/garch_baseline.py` | Walk-forward GARCH(1,1) with the `arch` package, monthly refits from 1970 -> `results_1970/market_forecasts.csv` |
| `fe_project/old/scipy_garch/stat_forecasts.py` | GARCH, GJR, GJR-t forecasts -> `results_1970/forecasts/*_stat.csv` |
| `fe_project/calibrate_c.py` | c so that mean exposure = 1 on 1970-1999, cap 2 -> `results_1970/c_values.json`, `weights.csv` |
| `fe_project/backtest.py` | Monthly simulation with drift and 5 bp per unit of turnover -> `results_1970/backtest_metrics.csv` |
| `fe_project/old/scipy_garch/evaluate.py` | Accuracy (QLIKE, Diebold-Mariano) and Sharpe tests -> `results_1970/eval_*.csv` |

**Result (2000-2024, `results_1970/eval_backtest.csv`, `eval_accuracy.csv`):** net Sharpe unmanaged 0.45, MM on RV 0.53,
GARCH 0.53, GJR 0.50. QLIKE: RV 0.47, GARCH 0.30, GJR 0.28 (GARCH and GJR far more accurate, DM p < 0.001).
**So: much better forecasts, zero Sharpe gain** (GARCH vs RV dSharpe -0.00, p 0.99). Drawdown -50% -> -19 to -23%.
[This is the puzzle the rest of the project explains.]

---

## Step 2. Why does decision-focused training not help? (exploratory, `fe_project/dfl_diagnosis/`)

**Question.** If accuracy does not translate into Sharpe, train the forecast on the portfolio objective instead.

[Key idea, the decomposition: SR_timed / SR_unmanaged = MT x VT with exposure normalised to mean 1.
 MT (mean timing) = E[xR] / E[R]: do you hold more in months that pay more? VT (variance timing) = sd(R) / sd(xR): do you cut risk?
 Better forecasts can only raise VT. If MT falls when VT rises, Sharpe does not move.]
[Theory: with a constant conditional mean, the Sharpe-optimal exposure is proportional to 1 / E[sigma^2 | information], so a
 decision loss has the same population target as QMLE. DFL can only help through misspecification, a time-varying risk-return
 relation, or frictions. A utility loss also adds the noise of realised returns: it should overfit.]

| File | What it does |
|---|---|
| `dfl_diagnosis/decomp.py` | MT x VT split for the market strategies |
| `dfl_diagnosis/decision_garch.py` | GARCH / GJR trained on the decision loss -> `results_1970/forecasts/*_dec.csv` |
| `dfl_diagnosis/risk_dfl.py` | 16 series, GARCH and GJR trained three ways: qmle, util (portfolio utility), risk (mean rv*w^2) -> `dfl_diagnosis/panel/` |
| `dfl_diagnosis/panel_eval.py` | Panel table and paired block-bootstrap tests |
| `dfl_diagnosis/mc_theory.py` | Monte Carlo of the theory (correct GARCH, GJR truth, positive risk-return slope) |
| `dfl_diagnosis/cost_dfl.py`, `cost_eval.py`, `cost_partial.py` | Cost-aware loss vs a simple partial-adjustment rule |
| `dfl_diagnosis/pooled_curvature.py` | Exposure curvature w = c * v^-p, per series vs pooled |
`dfl_diagnosis/README.md` gives the run order.

**Results (market 2000-2024, panel of 16):**
- VT rises with accuracy (RV 1.21, GARCH 1.25, GJR 1.27) while MT falls (1.02, 0.98, 0.91): the most accurate model has the
  worst Sharpe.
- Utility-trained models: MT 1.00-1.04 in sample, 0.90-0.91 out of sample (overfitting).
- Panel: the utility loss worsens variance timing in 16/16 series, no Sharpe gain; the risk loss equals QMLE.
- Monte Carlo: utility loss lowers Sharpe when the model is right; helps only with a positive risk-return slope.
- Costs: a cost-aware loss beats QMLE at 50 bp, but ties a simple partial-adjustment rule. [Frictions belong in the trading rule.]

---

## Step 3. Is the forecast the bottleneck? Better information (exploratory, `fe_project/exo_indicators/`, items 1-12 of its README)

**Question.** Maybe GARCH is missing information: options (VIX), credit spreads, news, uncertainty (EPU).
[If the binding constraint is MT, more variance information should raise accuracy and VT, but not Sharpe.]

| File | What it does |
|---|---|
| `exo_indicators/fetch_indicators.py` -> `indicators_monthly.csv` | Credit/term spreads (FRED), EPU, EMV, SF Fed news sentiment, monthly, real-time lags |
| `exo_test.py`, `exo_robust.py` | Volatility forecasts with one indicator at a time; OOS R2, Clark-West, event study of 12 largest surprises |
| `har_vrp.py` | HAR, HAR+GARCH, VIX forecasters; categorical EPU; VRP and nine-predictor return rules |
| `exo_mean.py`, `epu_check.py`, `epu_panel.py`, `combo_panel.py` | Indicators on the return side: w = c * max(mu_hat, 0) / v_hat |
| `crash_rule.py`, `bad_vol.py`, `regime_factor.py`, `xfactor.py` | Crash override, downside vol, regime rule, two-layer factor portfolio |

**Results:** VIX raises accuracy a lot (QLIKE 0.30 -> 0.26-0.28) but net Sharpe falls 0.71 -> 0.63-0.65 because MT falls
0.90 -> 0.78-0.81. Credit spreads, EPU, news: OOS R2 of volatility about 0 to +1%, no Sharpe gain. Return-side rules: small,
not significant (combination +0.04 on the panel). [Conclusion: the problem is mean timing, which is a return-prediction problem.]

---

## Step 4. A neural GARCH (GINN) (exploratory, `fe_project/ginn/`)

**Question.** Does a flexible model (GJR + neural network, with and without VIX) forecast better and time better?

| File | What it contains |
|---|---|
| `ginn/GINN_v2.ipynb` | Torch GJR / GJR-X (exact FFT filter + L-BFGS, matches scipy), GINN and GINN-VIX ensembles, weighted-loss variants, DM tests |
| `ginn/outputs/ginn_v2_daily.csv`, `ginn_v2_grid.csv` | Daily forecasts on the 2020-2026 test period; hyper-parameter grid |
| `ginn/figures/` | Architecture diagram |

**Results (test 2020-2026, daily loss):** GINN vs GJR t = -1.20, GINN-VIX vs GJR-X t = -1.36 (not significant); GJR-X vs GJR
t = -2.17 (VIX is the only real gain). Volatility-weighted loss: worse (t +2.15). [Flexibility does not fix MT.]
Not converted to monthly Sharpe (still on the to-do list).

---

## Step 5. Fixing the return side of the rule (exploratory, `fe_project/exo_indicators/` items 13-20, 2000-2024)

**Question.** If the leak is mean timing, can a rule stay invested when expected returns are high (rebounds)?

| File | Idea | Result |
|---|---|---|
| `fe_project/old/scipy_garch/garch_x.py`, `rules.py` | VIX^2 instead of RV in MM | best VT (1.28), worst MT (0.90): SR 0.51 vs 0.53 |
| `har_vrp.py` | Split VIX^2 = expected variance + VRP; VRP as return signal | MT 0.89 -> 1.24 but VT 0.89; SR 0.67 vs 0.70 |
| `xu_conditional.py` | Xu (2024) switches; conditional quintile targeting | +0.06 to +0.08 on the market, not significant |
| `icr_test.py` (+ `hkm_capital_ratio_monthly.csv`) | Intermediary capital ratio as return signal | +0.12 at lag 2, flips at lag 1, panel 0 |
| `rebal_speed.py` | Weekly / daily rebalancing | worse (daily 5-day -0.40) |
| `vix_term.py` (+ `vix_term_structure_monthly.csv`) | VIX vs VIX3M inversion | staying out during inversion is right; overrides hurt |
| `recovery_state.py` | **Recovery state** (vol high but falling), spec fixed before running, tested on untouched 1975-2007 | recovery months pay 1.69%/month, yet MM holds 0.21 exposure; rule +0.09 (n.s.) |
| `kalman_mu.py`, `vrp_kalman.py`, `vrp_garchx.py` | Kalman filters for "when to trust what" | reduce to MM / fragile |

[Takeaway: you can buy MT, but you pay in VT. **The mechanism** (exposure is cut in high-volatility months that are often the
best recovery months) **replicates out of sample**: this is the anatomy of the MT leak.]

---

## Step 6. Main horizon 1998-2025, the market and 16 series (now in final-version: scripts 1 and 2)

**Question.** The professor said vol timing no longer works. Test 1998-2025 with sub-periods 1998-2015, 2016-2025, 2021-2025.
[All parameters from pre-1998 data; same rule on all 16 series, each with its own real-time GARCH.]
First version `fe_project/main_eval/market_1998.py`, `panel_1998.py` (old settings). **Final version:**

| File | What it does |
|---|---|
| `1_us_market.py` | Market: unmanaged, MM on RV / GARCH / GJR / VIX^2, Xu, conditional, recovery. Sharpe, Sortino, MT, VT, drawdown, dSR with CI, Cederburg |
| `2_us_each_series.py` | All 16 series; mean gain with blocks common to all series; series that gain; per-series tables |
| `core/garch.py` | Real-time GARCH(1,1) / GJR, refit every January, exact FFT filter; forecasts cached in `results/cache/` |
| `core/rule.py` | w = c / v, c re-set every January (vol match, past months only), cap, costs |
| `core/variants.py` | Xu switches, conditional quintiles, recovery state |

**Final results (`results/main/1_us_market.txt`, `2_us_each_series.txt`):** market MM on GARCH 0.61 vs 0.52 unmanaged
(+0.09 [-0.13, +0.33]); drawdown -25% vs -50%; Cederburg +0.00. GJR: same Sharpe, better forecasts. 16 series: mean gain +0.00,
7 of 16 gain. After 2016 the market gain is +0.00. [The professor's claim holds.]

---

## Step 7. Factor mixes and combinations (final-version: scripts 3, 4, 5)

**Question.** If timing alone does not work, does allocation across factors, with or without timing, do better?

| File | What it does |
|---|---|
| `3_us_factor_weighting.py` | EW, inverse variance, min variance, tangency; each with and without timing the whole mix; EW of vol-managed factors |
| `4_us_factor_combos.py` | All 63 equal-weight subsets ranked; real-time factor picks; ranks before / after 1998; industry groups |
| `5_us_main_result.py` | Nine EW mixes timed with MM on the mix's own GARCH vs unmanaged; Sharpe, Sortino, CAGR, drawdown, levered to 15%; Cederburg. Writes the dashboard data `results/main/5_us_main_result.json` |

**Final results:** EW of six factors 0.71 vs market 0.52 (allocation does the work). Real-time factor picking fails.
Mkt + RMW + CMA + Mom is #1 of 63 in 1998-2025 (0.85) [picked with hindsight; only #14 before 1998]. Timed: 1.04 vs 0.85,
+0.19 [-0.05, +0.42], Sortino 1.82 vs 1.33, drawdown -6% vs -16%; after 2016 +0.03. Not significant after correction (Step 10).

---

## Step 8. International out-of-sample (final-version: scripts 6 and 7)

**Question.** Is the US result real? Test on data we had never seen, with the same rule and settings.
[Two independent sources: JKP (close analogues: ope_be ~ RMW, at_gr1 ~ CMA, ret_12_1 ~ Mom) and Ken French regional factors
(exact definitions). Data were fetched in the browser and checksum-verified.]

| File | What it contains |
|---|---|
| `data/jkp/<region>_{daily,monthly}.csv` | JKP factors mkt, ope_be, at_gr1, ret_12_1; 8 regions (world ex US, JPN, GBR, DEU, FRA, CAN, AUS, emerging), from 1986 |
| `data/ff_intl/<region>_{daily,monthly}.csv` | French 5 factors + Mom; Developed, Developed ex US, Europe, Japan, Asia Pacific ex Japan, North America; Jul 1990 on |
| `6_intl_jkp.py` | 8 regions, 6 mixes, pooled test over 7 non-overlapping regions, Cederburg |
| `7_intl_french.py` | 6 regions + US, 7 mixes, whole-mix vs leg-wise timing, pooled over Europe, Japan, Asia Pacific |

**Final results:** JKP pooled Mkt + Prof + Inv + Mom +0.05 (1998-2015 +0.13, 7/7 regions; after 2016 -0.09, 0/7).
Momentum alone +0.30 [+0.14, +0.48], 7/7, and +0.17 [+0.02, +0.32] after 2016. French: RMW + CMA + Mom +0.27 [+0.04, +0.49],
momentum +0.34. [The US mix result replicates only before 2016; momentum is the robust exception (Barroso and Santa-Clara 2015).]
Leg-wise timing (market unmanaged) is post-hoc and flips between cap 1.5 and cap 2: not a finding.

---

## Step 9. Robin's decision-focused experiment (final-version: script 9 and `robin_experiment/`)

**Question.** The direct DFL test: same model, objective A (QMLE) vs objective B (portfolio utility), for GARCH, GJR and the
Zhao et al. (2024) GARCH-LSTM (5 seeds), on the six French factors, real time from 1937 (1974 for RMW, CMA).

| File | What it contains |
|---|---|
| `robin_experiment/` | Robin's code (`backtest.py`, `backtest_zhao.py`, `models.py`, `train.py`, `evaluate.py`, `compare.py`), his tables in `results/`, his report `experiment1_report.pdf`, his README; `NOTE.md` explains what was copied |
| `robin_experiment/convert_pickles.py` | Turns his saved panels into `data/robin/<factor>_forecasts.csv` (checked: identical output) |
| `data/robin/<factor>_forecasts.csv` | Per month: f (factor return), rv, V_<model> (his real-time forecast), c_<model> (his scaling) for RV, GARCH A/B, GJR A/B, LSTM A/B, Zhao A/B seeds 0-4 |
| `9_dfl_robin.py` | Part 0 reproduces his reported Sharpe ratios (max difference 0.0005); Part 1 his forecasts under OUR rule and tests; Part 2 forecast accuracy A vs B |

[What I verified: his code has no look-ahead (refit each January on past data, c from training months, early stopping inside
training data); his reported numbers are reproduced exactly from his saved forecasts; his GARCH forecasts correlate 0.995-0.9999
with our independent pipeline. Not verified: the neural-network training itself (1.5 h run); ask Robin to re-run once.]

**Results (`results/main/9_dfl_robin.txt`), 1998-2025 under our protocol:** decision training (Zhao B - A) +0.10 to +0.12 on HML,
RMW, CMA, about 0 on Mkt, SMB, Mom; none significant, nothing survives Romano-Wolf among his 36 tests. For plain GARCH and
GJR it gives no consistent gain (mixed signs, none significant; on his own protocol it never helped them). The B forecasts are 1.2 to 3.5 times worse in QLIKE. [Decision training buys mean timing by giving up accuracy: the same
MT / VT trade-off as Steps 1-5, found by a different method.] On his own protocol (no cap, his c) the RMW gain was +0.21;
with our cap 1.5 it shrinks to +0.12 [-0.03, +0.27].

---

## Step 10. Protocol frozen, final run, multiple testing (final-version: SPEC.md, config.py, script 8, run_all.py)

**Why.** Each script had its own costs, c rule and blocks, and we had tried ~20+ variants. [Freeze the rules first, then run
once; report the old settings as robustness, so the choice cannot be accused of following the results.]

| File | What it does |
|---|---|
| `SPEC.md`, `config.py` | The frozen protocol; three specs: main (10 bp, c re-set yearly, cap 1.5), old (5 bp, frozen c, cap 2), uncapped |
| `core/stats.py` | Sharpe, Sortino, drawdown, circular block bootstrap, Cederburg combination, Holm, Benjamini-Hochberg, Romano-Wolf |
| `8_multiple_testing.py` | Every timed-vs-unmanaged comparison of scripts 1-7 and 9 on 1998-2025 in one family, one joint bootstrap |
| `run_all.py` | Runs 1-7, 9, then 8 for a spec; results in `results/<spec>/` |

**Final result:** 296 comparisons; 41 nominally significant at 5% (about 15 expected by luck); 1 survives Romano-Wolf (momentum,
JKP France). Old and uncapped specs: same picture (3 survivors, all momentum or momentum mixes in Europe).

---

## The story in one paragraph

Better volatility forecasts (GARCH, GJR, VIX, neural networks) improve variance timing but lower mean timing, because the
high-volatility months that the rule avoids include the best recovery months. Training the forecast for the decision instead
(Robin) moves along the same trade-off: worse forecasts, slightly better timing on weak factors, nothing robust. So volatility
timing reliably halves drawdowns but adds little Sharpe, and the small gain it had before 2016 disappeared afterwards in the US
and in 14 international regions. Momentum is the exception, where volatility spikes are crashes rather than rebounds.

## Verification checklist for tomorrow

1. `cd final-version && pip install -r requirements.txt && python run_all.py` (first run 15-20 min: fits every GARCH).
2. Compare with the committed logs in `results/main/` (they should match to the second decimal):
   - `1_us_market.txt`, 1998-2025: MM on GARCH Sharpe 0.61, unmanaged 0.52, MaxDD -25.1 vs -50.3.
   - `5_us_main_result.txt`: Mkt + RMW + CMA + Mom, Sharpe MM 1.04, unm 0.85, timing gain +0.19.
   - `6_intl_jkp.txt`, SUMMARY: Mom alone 1998-2025 +0.30, 7/7.
   - `9_dfl_robin.txt`, Part 0: largest difference from his report 0.0005.
   - `8_multiple_testing.txt`: 296 comparisons, Romano-Wolf 1.
3. `python run_all.py --spec old` should give our earlier numbers (e.g. US mix MM 1.09, JKP momentum +0.36).
4. For Steps 1-5 open the files listed above in `fe_project/`; they are exploratory and use the earlier settings.
