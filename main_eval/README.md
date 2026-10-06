# main_eval

Main evaluation of the project: 1998-2025, with sub-periods 1998-2015, the last 10 years (2016-2025) and the last 5 years (2021-2025).
Every parameter is estimated on data before 1998 and then frozen (c on 1970-1997; VIX-based rules on 1990-1997).
Run from the project root. Re-run and check every number before using it.

1. `python main_eval/market_1998.py`: market (Mkt-RF). Unmanaged, Moreira-Muir on RV / GARCH / VIX^2, Xu switches,
   conditional quintile targeting, recovery state, VRP Kalman. Net Sharpe, MT, VT, return, vol, max drawdown,
   paired block-bootstrap dSR vs unmanaged and vs Moreira-Muir on RV.
2. `python main_eval/panel_1998.py`: the same rules (minus the VIX ones) on all 16 series: market, SMB, HML, RMW, CMA, Mom and
   10 industries. Each series gets its own real-time GARCH(1,1) (re-estimated every January, FFT filter), cached in
   `main_eval/results/garch_<series>.csv` (about 3 minutes the first time; delete the cache to re-estimate).
   Mean net Sharpe for all 16 / factors / industries, mean dSR with common-block bootstrap, series better;
   per-series table in `main_eval/results/panel_1998_by_series.csv`.
3. `python main_eval/factor_mix_1998.py`: mixes of the six factors. Allocation: EW, inverse variance, minimum variance
   (GARCH variances + 60-month correlations), long-only tangency (expanding means); each with and without timing of the whole
   mix (c / portfolio variance), plus EW of vol-managed factors; vs the market alone and EW.

4. `python main_eval/subsets_1998.py`: (A) smaller factor mixes: all 63 equal-weight subsets as context, the hindsight pick
   RMW + CMA + Mom, and real-time selections (best 3 before 1998; top 3 re-chosen every January, expanding or trailing 5 years);
   (B) industry groups, defensive (NoDur, Hlth, Utils, Telcm) vs cyclical (Durbl, Manuf, Enrgy, HiTec, Shops, Other) and all 10,
   equal-weight portfolios under the main_eval rules.

5. `python main_eval/mix_curves_1998.py`: equal-weight factor mixes with Moreira-Muir on a GARCH(1,1) of the mix itself, and unmanaged;
   Sharpe, Sortino, CAGR, volatility, max drawdown for each period, as is and levered to 15% vol (leverage set on 1970-1997).
   Writes `main_eval/results/mix_curves_monthly.csv` (monthly net excess returns, for equity / drawdown plots).

6. `python main_eval/intl_mix_1998.py`: international out-of-sample check of item 5 on JKP Global Factor Data (jkpfactors.com,
   vw_cap, USD excess returns, files in `data/jkp/`, downloaded 6 Oct 2026). Regions: world ex US, Japan, UK, Germany, France,
   Canada, Australia, emerging. JKP analogues: mkt, ope_be (~RMW), at_gr1 (~CMA), ret_12_1 (~Mom). Same rule as item 5; GARCH and c
   use only pre-1998 data (calibration from 1990, later where data start later). Timing gain dSR with block-bootstrap CIs per region
   and pooled over the 7 non-overlapping regions. Output log: `main_eval/results/intl_mix_1998_log.txt`.

7. `python main_eval/intl_ff_1998.py`: the same check on Ken French's regional factors (identical definitions to the US factors;
   files in `data/ff_intl/`, July 1990 - Aug 2026, downloaded 6 Oct 2026): Developed ex US, Europe, Japan, Asia Pacific ex Japan,
   North America, Developed, plus the US for reference. GARCH from 1993, c on 1993-1997 (US: 1970 / 1970-1997). Three ways to time
   each mix: whole mix (one GARCH, as in 5 and 6), legs with the market unmanaged, all legs managed (the leg-wise designs were chosen
   after seeing item 6: post-hoc). Pooled test over Europe + Japan + Asia Pacific ex Japan. Log: `main_eval/results/intl_ff_1998_log.txt`
   (about 5 minutes the first time).

`garch_tools.py` holds the shared real-time GARCH (used by 2 to 7). If a fit lands on phi = 1 (integrated GARCH) the monthly forecast is
21 x the one-day variance instead of NaN (happened once, North America market, from 1997).
