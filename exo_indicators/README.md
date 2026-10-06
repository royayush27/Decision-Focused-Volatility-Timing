# exo_indicators

Do exogenous indicators (credit spreads, news attention, news sentiment) warn of volatility spikes early,
using only information available at each month-end? Market (Mkt-RF) plus a 16-series panel for the one lead.
Exploratory: re-run and check every number before using it.

## Timing (no look-ahead)
Row t is the month being forecast; the forecast is made at the end of t-1.
Credit and term spreads, EPU: value for month t-1 (monthly average of daily data, known at the end of t-1).
EMV and SF Fed news sentiment: value for month t-2 (both are published days to a week late). `_L1` versions use t-1 as a sensitivity check.
The EPU level is normalised by its authors over a fixed base period, which only shifts log EPU by a constant (absorbed by the intercept).
The news indices can be revised by their publishers; this is a caveat, not something the scripts can fix.

## Run order (from the project root)
1. `python exo_indicators/fetch_indicators.py` rebuilds `indicators_monthly.csv` from FRED and the SF Fed and compares it with the committed file.
2. `python exo_indicators/exo_test.py`: log RV forecasts, baseline = OLS on log GARCH + log lagged RV, expanding window, plus one indicator at a time.
   Reports OOS R2 vs baseline, Clark-West t, QLIKE, forecast revisions in GARCH-surprise spike months, VR, MT, net Sharpe with a block bootstrap.
   Panel A: spreads, estimation from 1970, evaluation 2000-2024. Panel B: news era with VIX as benchmark, evaluation 2005-2024.
3. `python exo_indicators/exo_robust.py`: end-of-month spread, looser timing, attention-weighted sentiment, and an event study of the 12 largest GARCH surprises.
4. `python exo_indicators/exo_mean.py`: indicators on the mean side, w = c * max(mu_hat, 0) / v_hat.
5. `python exo_indicators/epu_check.py`: sub-periods and timing variants for EPU on the mean side.
6. `python exo_indicators/epu_panel.py 1` (and `2`, `1 alt`, `2 alt`): EPU mean-side rule on all 16 series.
   Needs `dfl_diagnosis/panel/*_garch.csv`, i.e. run `dfl_diagnosis/risk_dfl.py` first.
7. `python exo_indicators/har_vrp.py`: HAR (day/week/month squared daily returns, log form), HAR+GARCH, VIX and
   HAR+GARCH+VIX forecasters; categorical EPU (trade, monetary, national security, lag t-2) on the vol side;
   vol + expected-return rules w = c*max(mu_hat,0)/v_hat with mu_hat from VRP, EPU or a nine-predictor combination.
8. `python exo_indicators/combo_panel.py`: the vol + expected-return rules on all 16 series (needs dfl_diagnosis/panel).
9. `python exo_indicators/crash_rule.py`: what happens the month after a crash (1963-2025), and a 'never hold less than 1
   after a crash / in a bear market' override on Moreira-Muir, market and 16 series, original 1970-1999 / 2000-2024 protocol.

Inputs used from the project: `results_1970/market_forecasts.csv`, `data/ff_daily_garchdata_long.csv`, `data/ff_monthly_rf.csv`, `data/ff_monthly_baselinedata.csv`, `data/VIXCLS.csv`,
`backtest.py`, `calibrate_c.py`, `dfl_diagnosis/risk_dfl.py`.
10. `python exo_indicators/bad_vol.py`: downside-volatility timing (Wang-Yan) and a 12-month trend overlay (cash when the
    past 12-month return is negative), market and 16 series, 1970-1999 / 2000-2024 protocol.
11. `python exo_indicators/regime_factor.py`: the GitHub regime rule on French factors, same-month VIX (look-ahead) vs lagged VIX.
12. `python exo_indicators/xfactor.py`: two-layer factor portfolio (timing x allocation): EW, EW vol-managed, inverse variance,
    minimum variance (GARCH variances + 60-month correlations), timed minimum variance; and QMLE vs decision-trained forecasts inside each.
13. `python exo_indicators/xu_conditional.py`: rule-design fixes for mean timing, no new data. Xu (2024) switches on Moreira-Muir
    (cash if the past 12-month mean is negative; vol-manage only if the 60-month risk-return intercept is positive) and
    conditional volatility targeting (Bongaerts, Kang, van Dijk 2020: scale only in the top / bottom quintile of lagged variance).
    Market (RV and GARCH) and 16-series panel (RV), 1970-1999 / 2000-2024 protocol.
14. `python exo_indicators/icr_test.py`: intermediary capital ratio (He, Kelly, Manela) as an expected-return signal.
    Data `hkm_capital_ratio_monthly.csv` (from He_Kelly_Manela_Factors_monthly_250627.csv on zhiguohe.net, 1970-01 to 2025-05,
    duplicate 2025-01 row resolved to the later one; checksums verified at download). ICR lag 2 (main), 1 and 4 (robustness).
    A: w = c * max(mu_hat, 0) / v_hat with mu_hat from log ICR; B: Moreira-Muir that never cuts below 1 when ICR is in its bottom quintile.
15. `python exo_indicators/rebal_speed.py`: Moreira-Muir rebalanced monthly, weekly or daily with 21-day or 5-day realised variance
    (market, daily data), 5 and 10 bps; MT / VT on daily returns and exposure right after the worst days.
16. `python exo_indicators/vix_term.py`: VIX term structure (VIX vs VIX3M, from Dec 2007; data `vix_term_structure_monthly.csv`
    from FRED VIXCLS and VXVCLS). Next-month returns by state, 'no cut while inverted' and 'back in after the panic' overrides,
    and a mu rule on log(VIX / VIX3M).
17. `python exo_indicators/recovery_state.py`: out-of-sample test of the 'recovery phase' idea from vix_term.py, with realised
    volatility only: high = last month's RV in its top 20%, recovery = high and 5-day RV < 63-day RV, panic = high and 5-day >= 63-day.
    Rule: Moreira-Muir with exposure at least 1 in recovery months. Specification fixed before running; c on 1965-1974,
    evaluation 1975-2007 (out of sample), 2008-2024 as an in-sample check; market and 16 series.
18. `python exo_indicators/kalman_mu.py`: Kalman filter for a time-varying expected return (random-walk mu, noise variance =
    kappa * last month's RV, so crash months get little weight). q and kappa by maximum likelihood on 1965-1999, then fixed;
    w = c * max(mu_hat, 0) / v, evaluation 2000-2024, market and 16 series. 'naive' fixes kappa = 1.
19. `python exo_indicators/vrp_kalman.py`: Kalman filter on the VRP predictive regression (time-varying a_t, b_t), noise
    kappa * GARCH variance; kappa and q by maximum likelihood on 1990-2004, evaluation 2005-2024, vs historical mean and static OLS VRP.
    Robustness over the prior and a 16-series panel.
20. `python exo_indicators/vrp_garchx.py`: full decomposition, GARCH-X in the denominator and the VRP left over after GARCH-X
    (VIX^2 - GARCH-X) in a precision-weighted Kalman regression for mu; vs vol timing on GARCH / GARCH-X and VRP-Kalman on GARCH.
    Needs old/scipy_garch/forecasts_vix/garchx_vix_stat.csv.
