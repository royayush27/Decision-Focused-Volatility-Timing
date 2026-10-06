# Protocol (frozen 6 Oct 2026, before the final run)

Every script reads these settings from `config.py`, so no script can disagree with another. Each choice follows
Moreira and Muir (2017), Cederburg et al. (2020) or Robin's Experiment 1, not what made a result look better. Our earlier
settings and the uncapped rule are kept as robustness specs; a result that changes between specs is reported as such.

## 1. Trading rule (Moreira-Muir on GARCH)

| Item | Main setting | Why |
|---|---|---|
| Exposure | w_t = c_t / v_t, with v_t the real-time GARCH(1,1) forecast of month t's variance | Moreira and Muir (2017), with a model forecast instead of last month's realised variance |
| GARCH | Zero-mean Gaussian QMLE on daily returns, re-estimated every January on all data before it (expanding window); monthly forecast = sum of the 1- to 21-day-ahead forecasts made on the last trading day of t-1 | Real time: only information up to t-1 |
| Scaling c | Re-set every January on all past months with a forecast (at least 24): c is chosen so that the managed series, with the cap applied, has the same volatility as the unmanaged one over those months | Moreira-Muir convention, applied in real time as in Cederburg et al. (2020) and Robin; 24 months is the shortest window that lets every international region trade from 1998 |
| Leverage cap | w in [0, 1.5] | Moreira and Muir (2017), Table V |
| Trading cost | 10 bp per unit of exposure change (measured after the month's drift) | Moreira and Muir (2017), Table IV |
| Rebalancing | Monthly | |

Mixes are equal-weight portfolios of their factors, rebalanced monthly; the rule is applied to the mix as one asset, with a
GARCH fitted on the mix's own daily returns. Rule variants (Xu switches, conditional quintiles, recovery state) are built on
the same exposure and capped again (`core/variants.py`).

## 2. Samples

| Item | Setting |
|---|---|
| Main evaluation | January 1998 to December 2025 |
| Sub-periods | 1998-2015, 2016-2025 (last 10 years), 2021-2025 (last 5 years) |
| US data | Ken French library, daily and monthly, July 1963 on; GARCH refits from January 1970 |
| International | JKP Global Factor Data (8 regions, from 1986) and Ken French regional factors (6 regions, from July 1990); GARCH refits from 1990-1996 depending on the data start (two full years of daily data first) |

## 3. Tests

| Test | Setting |
|---|---|
| Sharpe difference | Timed minus unmanaged net Sharpe ratio; circular block bootstrap, **12-month blocks**, 5,000 draws, 95% percentile interval, two-sided p-value |
| Pooled tests | Mean difference across series or regions, with the same resampled months for every series |
| Cederburg et al. (2020) | Each month, mean-variance weights on (unmanaged, timed) estimated on past months only (at least 120), applied to the next month; combination Sharpe minus unmanaged Sharpe, same bootstrap |
| Multiple testing | Every timed-vs-unmanaged comparison of scripts 1-7 on 1998-2025 in one family, one joint bootstrap; Holm, Benjamini-Hochberg and Romano-Wolf step-down p-values (`8_multiple_testing.py`) |

## 4. Specs

| Spec | Cost | c | Cap | Use |
|---|---|---|---|---|
| `main` | 10 bp | re-set every January (volatility match) | 1.5 | all headline numbers |
| `old` | 5 bp | frozen once: mean exposure 1 on the pre-1998 window (US 1970-1997) | 2 | our earlier settings |
| `uncapped` | 10 bp | re-set every January | none | comparable with Moreira-Muir and Robin |

The tests (blocks, draws, Cederburg window) are the same in every spec. Run with `python run_all.py --spec <name>`;
results go to `results/<spec>/`.
