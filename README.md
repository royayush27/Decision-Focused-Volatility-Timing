# Decision Focused Volatility Timing

**Introduction to Financial Engineering, Hanyang University**
Supervisor: Prof. Song Jae Wook (Introduction to Financial Engineering)
Team: Ayush, Farres, Robin, Zheru

> **Status (Sep 2026):** Data collection is done and validated. Next up is the first model: GARCH(1,1) on daily market returns, turned into a monthly variance forecast and compared against realized variance.

---

## Contents

1. [The question](#1-the-question)
2. [Two layers: course vs research](#2-two-layers-course-vs-research)
3. [Why the data looks the way it does](#3-why-the-data-looks-the-way-it-does)
4. [The dataset](#4-the-dataset)
5. [How to load and reproduce](#5-how-to-load-and-reproduce)
6. [What comes next](#6-what-comes-next)
7. [Open tasks](#7-open-tasks)
8. [References](#8-references)

---

## 1. The question

Volatility timing means scaling your exposure to an asset down when its expected variance is high and up when it is low. Moreira and Muir (2017) showed this raises Sharpe ratios for most factors. Cederburg et al. (2020) pushed back: once you remove look-ahead choices and account for costs, most of the gain disappears out of sample.

Every paper in this debate estimates the variance model one way (statistical fit, usually maximum likelihood) and then uses it for a completely different purpose (making money in a portfolio). Our question is whether that mismatch matters.

**Does estimating a variance model to maximize portfolio utility net of trading costs, instead of fitting variance well, produce better volatility-timed portfolios?**

### The experiment

We hold two things fixed and change the other two.

| Fixed | Varied |
|---|---|
| The data (same series, same dates) | The **variance model** |
| The trading rule: weight = c / forecast variance, rebalanced monthly | The **estimation objective** |

**Variance models**

| Model | What it is | Forecast for next month |
|---|---|---|
| Moreira-Muir RV | Sum of squared daily returns last month | `rv_lag1` directly |
| GARCH(1,1) | Classic conditional variance model | Fit on daily data, closed-form sum over the next ~21 days |
| GJR-GARCH with t errors | Adds asymmetry (bad news raises vol more) and fat tails | Same formula, persistence = α + δ/2 + β |
| GINN / GARCH-LSTM | GARCH combined with a neural net | No closed form, simulate ~21 days forward and average |

**Estimation objectives**

| Objective | Meaning |
|---|---|
| QMLE | Standard: pick parameters that fit the variance best |
| Decision focused | Pick parameters that give the best portfolio utility after trading costs |

If only the objective changes and results improve, we have evidence that *how* you estimate matters as much as *which* model you use.

---

## 2. Two layers: course vs research

The project has two separate jobs. They use different data and **are never merged**.

| | Course layer | Research layer |
|---|---|---|
| Purpose | Meet the course requirement | Answer the research question |
| Data | Professor's CRSP monthly stock file, 2000-2020 | Ken French Data Library, daily, 1963-2026 |
| Output | Monthly rebalanced stock portfolio vs S&P 500 | Volatility-timed factor and industry portfolios |
| Status | Two data checks pending (see [section 7](#7-open-tasks)) | Dataset built and validated |

The course layer is where the vol timing idea gets applied to the professor's stock data at the end. The research layer is where we actually test the models, because it needs daily data and a long history.

---

## 3. Why the data looks the way it does

This section answers "why didn't we just use X?" for every X we considered.

### 3.1 Why we dropped the 3,300-stock LSEG pull

The stock list came from a *current* screener. Every company that went bankrupt, got delisted or was acquired before today is missing. This is **survivorship bias**, and it is especially harmful for a volatility study. The firms that blew up are exactly the most violent volatility episodes in history. A model trained without them learns a world where nothing collapses.

It was also huge (on the order of 100 million data points) and was hitting API quotas. **That extraction should stop.**

### 3.2 Why we don't patch in dead stocks

We looked at adding delisted firms from other sources (scraped delisting lists, Datastream dead lists, matching CRSP to LSEG security by security). All of them fail for one of these reasons:

- The lists have names but no prices or returns.
- Tickers get recycled, so matching by name is unsafe.
- Different vendors define returns and adjustments differently. Splicing them creates artificial jumps, and a variance model will happily fit those jumps.

### 3.3 Why the Ken French library

| Reason | Detail |
|---|---|
| **It is the literature standard** | Moreira & Muir (2017), Cederburg et al. (2020) and Wang & Yan (2021) all test on these exact series. Our numbers are directly comparable to theirs. |
| **No survivorship bias** | French builds portfolios from CRSP, which includes dead firms and their delisting returns. Portfolios don't delist. |
| **Anyone can replicate** | Free, public, no login. A reader doesn't have to trust our data cleaning. |
| **Long daily history** | Around 15,900 trading days per series. |

**Honest caveat:** French data is built from CRSP, so the professor's CRSP file is *not* an independent out-of-sample test for it. Out-of-sample evidence comes from the design instead (see [section 6](#6-what-comes-next)).

### 3.4 Why daily and not monthly

GARCH needs a lot of observations. With 252 monthly points (2000-2020), maximum likelihood estimates are badly biased, and the bias pushes persistence (α + β) down. Lower persistence means a flatter forecast, which mechanically weakens the timing signal. So an unfair monthly fit would make GARCH look worse than it is.

**Rule: fit on daily returns, then aggregate to a monthly forecast. Never fit GARCH on monthly returns.**

### 3.5 Other choices

| Choice | Why |
|---|---|
| **Value-weighted** industries, not equal-weighted | EW portfolios overweight tiny stocks, whose daily prices bounce between bid and ask. That bounce inflates measured volatility even when nothing is happening. |
| **10 industries** (30 as a robustness check) | 10 is broad and clean. 30 gives more series later if needed. |
| **Excess returns everywhere** | Factors already come as excess or long-short returns. Industries come as total returns, so we subtract the risk-free rate. |
| **Monthly rebalancing, 1-month RV lookback** | Matches Moreira-Muir and the course requirement. |
| **LSEG parked** | Optional later as a vendor robustness check. Not on the critical path. |

---

## 4. The dataset

### 4.1 Source files

All three come from the [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

| File | Contents |
|---|---|
| `F-F_Research_Data_5_Factors_2x3_daily_CSV.zip` | Mkt-RF, SMB, HML, RMW, CMA, RF |
| `F-F_Momentum_Factor_daily_CSV.zip` | Mom |
| `10_Industry_Portfolios_daily_CSV.zip` | 10 industries (we use the value-weighted block) |

### 4.2 The 16 series

| Type | Series | What it is |
|---|---|---|
| Market | `Mkt-RF` | US market minus T-bill |
| Long-short factors | `SMB`, `HML`, `RMW`, `CMA`, `Mom` | Size, value, profitability, investment, momentum |
| Industries (long only) | `NoDur`, `Durbl`, `Manuf`, `Enrgy`, `HiTec`, `Telcm`, `Shops`, `Hlth`, `Utils`, `Other` | Value-weighted industry portfolios, minus RF |

Keep the distinction between long-short factors and long-only industries in mind. They behave differently when timed, and it gives us a natural split for robustness.

### 4.3 Coverage

| | |
|---|---|
| Span | 1963-07-01 to 2026-08-31 |
| Trading days | 15,897 |
| Daily rows (long format) | 254,352 (15,897 × 16) |
| Monthly rows | 12,128 |
| Units | Decimal returns (0.01 = 1%) |

### 4.4 Output files (in `datasets/`)

**Use the Parquet files.** The CSVs are kept for quick viewing in Excel.

**`ff_daily_garchdata.parquet`**: daily, long format. Input for all GARCH-family models.

| Column | Type | Meaning |
|---|---|---|
| `Date` | datetime | Trading day |
| `series` | string | One of the 16 series |
| `ret` | float | Daily excess return, decimal |
| `month` | Period[M] | Calendar month of the date |

**`ff_monthly_baselinedata.parquet`**: monthly, long format. Input for the Moreira-Muir baseline and for evaluating every strategy.

| Column | Meaning |
|---|---|
| `series` | One of the 16 series |
| `month` | Calendar month |
| `ret_m` | Monthly compounded return: ∏(1 + r_d) − 1 |
| `rv` | Realized variance: Σ r_d² over the month's trading days |
| `n_days` | Trading days in the month |
| `rv_lag1` | Last month's `rv`. **This is the Moreira-Muir forecast for this month.** |
| `incomplete` | True if `n_days` < 15 |
| `month_end` | Last trading date in the month |

Other files: `ff_merged.csv` (wide daily), `ff_long.csv`, `monthly_returns_rv.csv`.

### 4.5 Validation checks (all pass)

| Check | Expected | Got |
|---|---|---|
| Worst Mkt-RF day | Black Monday, 19 Oct 1987, about −17% | −17.44% |
| Daily Mkt-RF std | 0.9% to 1.2% | Pass |
| Dates unique and sorted | Yes | Pass |
| Annualized vol from RV vs from daily std × √252 | Should match | 16.21% vs 16.21% |
| First `rv_lag1` per series is empty | Yes (no prior month) | Pass |

### 4.6 Quirks to know about

- **September 2001 has 15 trading days** because markets closed after 9/11. It sits right at the `incomplete` threshold.
- **Momentum starts in 1926** but the 5-factor file starts in July 1963. The inner merge trims everything to the common window.
- **Missing values** in French files are coded as −99.99 or −999. They are converted to NaN on read.
- **French CSVs are not plain CSVs.** The industry file holds several stacked tables (VW returns, EW returns, firm counts, firm size). We read only the VW block (`nrows=26319`). If French updates the file, that number changes, so check the last date read if you re-download.

---

## 5. How to load and reproduce

### Load the data (what most people need)

```python
import pandas as pd

daily   = pd.read_parquet("datasets/ff_daily_garchdata.parquet")
monthly = pd.read_parquet("datasets/ff_monthly_baselinedata.parquet")

mkt = daily.loc[daily["series"] == "Mkt-RF"].set_index("Date")["ret"]
```

Why Parquet: it keeps column types (dates stay dates, `month` stays a Period), it is much smaller, and it loads faster. A CSV turns everything back into text.

### Rebuild from scratch

Run `Data_Collection.ipynb` top to bottom. It:

1. Downloads the three zips from Dartmouth into `raw/`
2. Reads each file with the right header and footer offsets
3. Merges on `Date` (inner join) and divides by 100 (French publishes percent)
4. Subtracts RF from the 10 industries
5. Runs the validation checks above
6. Reshapes to long format and builds the monthly table
7. Writes everything to `datasets/`

If Dartmouth blocks the download, fetch the zips in a browser and drop them into `raw/`.

---

## 6. What comes next

### Step 1: First GARCH forecast (in progress)

Fit GARCH(1,1) on daily Mkt-RF over a training window (for example 1963-1999). Turn the daily forecast into a monthly one with:

```
Var_t[R_month] = H·σ² + (v_{t+1} − σ²) · (1 − (α+β)^H) / (1 − (α+β))
where σ² = ω / (1 − α − β),  H ≈ 21 trading days
```

Check the formula against the `arch` package's own multi-step forecast. Note that `arch` works best with returns in percent, so fit on `ret * 100` and divide the variance by 100² afterwards. Then plot the GARCH monthly forecast against `rv_lag1`.

### Step 2: Moreira-Muir baseline, done properly

Weight = c / `rv_lag1`. Moreira-Muir choose c so the timed portfolio has the same volatility as the original **over the full sample**. That uses future information, and it is one of the specific things Cederburg et al. criticize. We set c using **training data only** and report both versions.

### Step 3: Walk-forward evaluation

Since CRSP is not independent of French, out-of-sample evidence comes from the design:

| Protection | How |
|---|---|
| **Time** | Expanding window. Fit on data up to month t, forecast month t+1, move forward. No future data ever enters a fit. |
| **Assets** | Develop on factors, then test on industries (or the reverse) without re-tuning. |
| **Vendor (optional)** | Re-run on LSEG index data as a robustness check. |

### Step 4: The full comparison

Run every model under both objectives, then compare Sharpe ratios, utility and turnover after trading costs.

---

## 7. Open tasks

| Task | Owner | Notes |
|---|---|---|
| Stop the 3,300-stock LSEG extraction | Farres | See [3.1](#31-why-we-dropped-the-3300-stock-lseg-pull) |
| CRSP check: is `ADJ_PRC` dividend-adjusted? | TBD | Compare a high-dividend stock against its published total return. If it only adjusts for splits, returns are understated by about 2% a year and the S&P 500 comparison is unfair. |
| CRSP check: is the universe point-in-time? | TBD | Count distinct PERMNO per month. A decline over 2000-2020 means dead firms are included (good). A flat count means the file was screened. |
| GARCH(1,1) fit and monthly aggregation | In progress | Step 1 above |
| Moreira-Muir baseline with training-only c | Next | Step 2 above |

---

## 8. References

- Moreira, A. and Muir, T. (2017). Volatility-Managed Portfolios. *Journal of Finance*, 72(4).
- Cederburg, S., O'Doherty, M. S., Wang, F. and Yan, X. (2020). On the Performance of Volatility-Managed Portfolios. *Journal of Financial Economics*, 138(1).
- Wang, F. and Yan, X. (2021). Downside Risk and the Performance of Volatility-Managed Portfolios. *Journal of Banking & Finance*, 131.
- Fama, E. F. and French, K. R. (2015). A Five-Factor Asset Pricing Model. *Journal of Financial Economics*, 116(1).
- Glosten, L., Jagannathan, R. and Runkle, D. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5).
- Kenneth R. French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html