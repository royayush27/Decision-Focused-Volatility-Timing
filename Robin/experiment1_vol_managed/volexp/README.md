# Experiment 1 — Decision-trained volatility models on the Moreira–Muir factors

Replicates the Moreira & Muir (2017) setting (French factors Mkt, SMB, HML, Mom, RMW, CMA; benchmark = the
unmanaged factor) and replaces the variance forecast with GARCH(1,1), GJR-GARCH and a GARCH-LSTM, each trained on
objective A (daily QMLE) or objective B (monthly utility of the managed portfolio after costs). Everything is run in
real time with an expanding window (Cederburg et al. 2020).

## Files
| File | What it does |
| --- | --- |
| `data.py` | Reads the French CSVs in `data/` (monthly + daily), percent → decimals |
| `mm_replication.py` | In-sample replication of MM Table I (full-sample c) |
| `models.py` | GARCH / GJR / GARCH-LSTM in PyTorch; exact FFT GJR recursion; month-ahead forecast; utility |
| `train.py` | Objective A (L-BFGS QMLE; Adam + early stopping for the LSTM) and objective B (Adam from the A solution, early stopping on the last 20% of training months, mean frozen) |
| `check_arch.py` | Checks our QMLE against the `arch` package |
| `backtest.py` | Real-time backtest: refit every January, forecasts at each month-end, c from past months only. `python backtest.py Mkt SMB ...`; `SEED_OFFSET=1000` reruns only the LSTM with another seed |
| `evaluate.py` | Out-of-sample tables: Sharpe, alpha, costs, turnover, block-bootstrap Sharpe tests, Cederburg combination, spike test |
| `seed_check.py` | First-round GARCH-LSTM (overlay) across two seeds |
| `backtest_zhao.py` | Update: Zhao et al. (2024) GARCH-LSTM (eq. 7, 126-day windows), 5 seeds per annual refit, A and B |
| `compare.py` | Update: 36 pre-specified tests on net Sharpe (joint block bootstrap; Holm, Benjamini-Hochberg, Romano-Wolf), Diebold-Mariano on QLIKE, seed spread |
| `results/` | All outputs (CSV tables, pickled monthly panels `bt_<factor>.pkl`, fitted parameters) |

## Key settings
- Daily data from the French library (CRSP vintage Aug 2026); first trade after 10 years of data.
- Month-ahead forecast = sum of the 21 daily variance forecasts.
- Costs 10 bp per unit of |Δw| (MM Table IV); γ = 5 in the utility.
- GARCH-LSTM: H_t = h_t^GJR · exp(g_t), g_t = tanh(linear(LSTM(8) over the last 63 days of z, |z|, log h)); output
  layer initialised at zero so the model starts at its GJR fit.

## Zhao et al. model: choices where the paper is silent
- w is an 8-vector with sum|w| < 1 (variance stays positive); hidden size 8; output layer starts at 0, i.e. at the GJR fit.
- Gaussian QMLE (their best was Student-t, v=5) so objective A is identical across models.
- Each 126-day window starts from the GJR-filtered variance of its first day; month-ahead forecast = GJR mean
  reversion from the model's one-day forecast.
- The recursion runs in float32 on percent units (matches float64 to 3e-7). Mkt and HML were run before this
  speed-up and before the one-pass early stopping in `fit_B` (same states evaluated); results are equivalent.

## Requirements
`pip install torch arch statsmodels pandas scipy`. The full backtest takes ~1.5 h on 2 CPU cores.
