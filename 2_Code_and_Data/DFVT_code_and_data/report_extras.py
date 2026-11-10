"""Extra tables and figures for the report that the notebook does not print: the extended performance table
(Table 4: CAGR, Calmar, CAPM alpha and beta, worst month, value of $1), the sub-period table (Table 5), calendar-year
returns, the value of $1 and the rolling 36-month Sharpe ratio, including the decision-trained GARCH-LSTM on the market.
Run from this folder after the data are in place:  python report_extras.py   (about one minute).
`python report_extras.py --paper` also redraws all notebook figures in the report's style into figures_paper/."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nb_loader import load_notebook

PAPER = "--paper" in sys.argv
g = load_notebook(paper_style=PAPER)
globals().update({k: v for k, v in g.items() if not k.startswith("__")})
OUT = Path("results"); OUT.mkdir(exist_ok=True)
A, B = PERIODS[MAIN]

# the decision-trained GARCH-LSTM on the market, from the same exported file as Section 6
W["LSTM Dec"] = lstm_files["Mkt"]["W_zhao_B_ensemble"].reindex(ret.index)
nets["LSTM Dec"] = backtest(W["LSTM Dec"].loc["1970-01":], mkt)
trades["LSTM Dec"] = turnover(W["LSTM Dec"].loc["1970-01":], mkt)
assert np.allclose(nets["LSTM Dec"].loc[A:B], dfl_nets[("Mkt", "LSTM Dec")].loc[A:B])
COLS = ["Buy and hold"] + TIMED_MAIN + ["LSTM Dec"]
u = nets["Buy and hold"].loc[A:B]


def capm(z, m, lags=5):
    """OLS of z on the market with Newey-West (Bartlett) standard errors."""
    X = np.column_stack([np.ones(len(m)), m]); y = np.asarray(z)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; T = len(y)
    Xe = X * e[:, None]; S = Xe.T @ Xe / T
    for j in range(1, lags + 1):
        G = Xe[j:].T @ Xe[:-j] / T; S += (1 - j / (lags + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T); V = Q @ S @ Q / T
    return b, b / np.sqrt(np.diag(V))


def extended(k, a=A, b=B):
    z, r_f = nets[k].loc[a:b], rf.loc[a:b]
    wealth = (1 + z + r_f).cumprod(); cagr = wealth.iloc[-1] ** (12 / len(z)) - 1; mdd = max_drawdown(z, r_f)
    coef, t = capm(z, mkt.loc[a:b])
    return {"CAGR %": 100 * cagr, "mean excess %": 1200 * z.mean(), "sd %": 100 * z.std() * 12 ** .5, "Sharpe": sharpe(z),
            "max DD %": 100 * mdd, "Sortino": sortino(z), "Calmar": cagr / abs(mdd),
            "max DD at market vol %": 100 * max_drawdown(same_vol_as(z, u), r_f), "CER %": cer(z), "beta": coef[1],
            "alpha %": 1200 * coef[0], "alpha t": t[0], "worst month %": 100 * z.min(), "value of $1": wealth.iloc[-1],
            "avg exposure": W[k].loc[a:b].mean(), "traded per year": 12 * trades[k].loc[a:b].mean()}


table4 = pd.DataFrame({k: extended(k) for k in COLS})
table4.to_csv(OUT / "table4_performance.csv")
print("Table 4\n", table4.round(3), "\ncash CAGR %:", round(100 * ((1 + rf.loc[A:B]).prod() ** (12 / len(u)) - 1), 2))

rows = {}
for p in ["1998-2015", "2016-2025", "2021-2025"]:
    a, b = PERIODS[p]
    for k in COLS:
        z, r_f = nets[k].loc[a:b], rf.loc[a:b]; w_ = (1 + z + r_f).cumprod()
        rows[(k, p)] = {"CAGR": 100 * (w_.iloc[-1] ** (12 / len(z)) - 1), "vol": 100 * z.std() * 12 ** .5,
                        "Sharpe": sharpe(z), "max DD": 100 * max_drawdown(z, r_f)}
table5 = pd.DataFrame(rows).T.unstack(1); table5.to_csv(OUT / "table5_subperiods.csv")
print("Table 5\n", table5.round(3))

STYLE = {"Buy and hold": ("Buy and hold (S&P 500 proxy)", "black", "-", 1.8), TIMED["RV"]: ("Timed, RV", COLORS["RV"], "-", 1.2),
         TIMED["GARCH"]: ("Timed, GARCH(1,1)", COLORS["GARCH"], "-", 1.2), TIMED["GJR"]: ("Timed, GJR-GARCH", COLORS["GJR"], "-", 1.2),
         TIMED["LSTM"]: ("Timed, GARCH-LSTM", COLORS["LSTM"], "-", 1.2), "LSTM Dec": ("Timed, GARCH-LSTM Dec", COLORS["LSTM"], "--", 1.2)}


def bands(ax):
    for s, e in CRISES.values():
        ax.axvspan(pd.Period(s, "M").to_timestamp(), pd.Period(e, "M").to_timestamp(how="end"), color="grey", alpha=0.15, lw=0)


fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
for ax, log in zip(axes, [True, False]):
    for k, (lab, col, ls, lw) in STYLE.items():
        z = nets[k].loc[A:B]; wealth = np.r_[1.0, (1 + z + rf.loc[A:B]).cumprod()]
        t = (as_time(z.index) + pd.offsets.MonthEnd(0)).insert(0, pd.Timestamp("1997-12-31"))
        ax.plot(t, wealth, color=col, ls=ls, lw=lw, label=f"{lab} (${wealth[-1]:.2f})")
    cash = np.r_[1.0, (1 + rf.loc[A:B]).cumprod()]
    ax.plot(t, cash, color="grey", ls=":", lw=1.2, label=f"Cash, T-bill (${cash[-1]:.2f})")
    bands(ax)
    if log:
        ax.set_yscale("log"); ax.set_yticks([0.5, 1, 2, 4, 8, 12]); ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylabel("value of $1" + (" (log scale)" if log else ""))
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
if not PAPER:
    axes[0].set_title("Value of $1 invested in January 1998, total return net of 10 bp costs (grey bands: crises)", loc="left", weight="bold")
plt.tight_layout(); plt.savefig(FIGURES / "fig_growth.png", dpi=170); plt.close()

years = pd.DataFrame({k: 100 * ((1 + nets[k].loc[A:B] + rf.loc[A:B]).groupby(nets[k].loc[A:B].index.str[:4]).prod() - 1) for k in STYLE})
years.to_csv(OUT / "calendar_year_returns.csv")
print("years in which each timed portfolio beat the market:", {k: int((years[k] > years["Buy and hold"]).sum()) for k in COLS[1:]})
fig, ax = plt.subplots(figsize=(13, 4.6)); x = np.arange(len(years)); wd = 0.85 / len(STYLE)
for i, (k, (lab, col, ls, lw)) in enumerate(STYLE.items()):
    ax.bar(x + (i - (len(STYLE) - 1) / 2) * wd, years[k], wd, color=col, label=lab, hatch="//" if ls == "--" else None,
           alpha=0.55 if ls == "--" else 1, edgecolor="white", lw=0.3)
ax.axhline(0, color="black", lw=0.8); ax.set_xticks(x, years.index, rotation=90); ax.set_ylabel("total return, %")
ax.legend(frameon=False, ncol=6, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0))
fig.suptitle("Calendar-year total returns, net of costs, 1998-2025", x=0.01, ha="left", weight="bold")  # no-op in paper style
plt.tight_layout(); plt.savefig(FIGURES / "fig_years.png", dpi=170); plt.close()

fig, ax = plt.subplots(figsize=(11, 4))
for k, (lab, col, ls, lw) in STYLE.items():
    z = nets[k].loc["1995-01":B]; r = (z.rolling(36).mean() / z.rolling(36).std() * np.sqrt(12)).loc[A:B]
    ax.plot(as_time(r.index), r, color=col, ls=ls, lw=lw, label=lab)
bands(ax); ax.axhline(0, color="grey", lw=0.8); ax.set_ylabel("Sharpe ratio, past 36 months")
ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower right")
if not PAPER:
    ax.set_title("Rolling 36-month Sharpe ratio, net of costs", loc="left", weight="bold")
plt.tight_layout(); plt.savefig(FIGURES / "fig_rolling_sharpe.png", dpi=170); plt.close()

z = nets["LSTM Dec"].loc[A:B]
print("LSTM Dec vs buy and hold: bootstrap", bootstrap_test(z, u), "Ledoit-Wolf", ledoit_wolf_test(z, u))
print("Done: results/ and figures/")
