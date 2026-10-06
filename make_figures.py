"""Figures for the README and the slides, built from results/main (run run_all.py first).
Every number in a figure comes from a table the scripts saved, so if your run differs, the figures follow your run.
Output: figures/fig1_us_market.png ... fig5_recovery_state.png
Usage: python make_figures.py            (README versions, with titles)
       python make_figures.py --slides   (same figures without the big title, in figures/slides/, for the deck)"""
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from core import data
from core.garch import garch_monthly
from core.rule import mm_weights
from core.variants import rv_states

R = config.ROOT / "results" / "main"
SLIDES = "--slides" in sys.argv
OUT = config.ROOT / "figures" / ("slides" if SLIDES else ""); OUT.mkdir(parents=True, exist_ok=True)
S = config.SPECS["main"]
BLUE, ORANGE, GREY, RED = "#2a78d6", "#eb6834", "#8a8984", "#e34948"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
                     "savefig.bbox": "tight", "legend.frameon": False})
NOTE = "Net of 10 bp per unit of turnover, exposure cap 1.5, c re-set every January in real time."


def parse_ci(s):
    """'+0.09 [-0.13, +0.33] (3/3)' -> (0.09, -0.13, 0.33)"""
    x = [float(v) for v in re.findall(r"[-+]?\d+\.\d+", s)[:3]]
    return tuple(x)


def title(obj, text, **kw):
    """Main figure title; left out of the slide versions (the slide title says it)."""
    if not SLIDES:
        (obj.suptitle if hasattr(obj, "suptitle") else obj.set_title)(text, **kw)


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("saved", OUT / name)


# ---------------------------------------------------------------- Fig 1: US market, growth, exposure, drawdown
N = pd.read_csv(R / "nets_1_us_market.csv", index_col="month")
u, m = N["US market | Mkt-RF | Unmanaged"], N["US market | Mkt-RF | MM on GARCH"]
e = data.us_monthly(["Mkt-RF"])["Mkt-RF"]
vg = garch_monthly("us_Mkt-RF", data.us_daily(["Mkt-RF"])["Mkt-RF"], config.US_FIRST_GARCH_YEAR).reindex(e.index)
w = mm_weights(vg, e, S)[0].loc[u.index[0]:u.index[-1]]
t = pd.PeriodIndex(u.index, freq="M").to_timestamp()
rfm = data.rf().reindex(u.index)
gu, gm = (1 + u + rfm).cumprod(), (1 + m + rfm).cumprod()       # total return (as in the MaxDD of the tables)
fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
ax[0].plot(t, gu, color=GREY, lw=1.6, label=f"Buy and hold (Sharpe {u.mean() / u.std() * 12 ** .5:.2f})")
ax[0].plot(t, gm, color=BLUE, lw=2, label=f"Volatility-timed, MM on GARCH (Sharpe {m.mean() / m.std() * 12 ** .5:.2f})")
ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of $1\n(total return, log scale)")
ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}"))
ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax[0].set_yticks([1, 2, 4, 8]); ax[0].legend(loc="upper left")
title(ax[0], "US stock market 1998-2025: timing halves the drawdown, adds little Sharpe", loc="left", color=INK, fontsize=12)
ax[1].fill_between(t, w.to_numpy(), color=BLUE, alpha=.25, lw=0); ax[1].plot(t, w.to_numpy(), color=BLUE, lw=1.2)
ax[1].axhline(1, color=GREY, lw=1, ls="--"); ax[1].axhline(S["cap"], color=INK2, lw=.8, ls=":")
ax[1].set_ylabel("Exposure\n(1 = buy and hold)"); ax[1].set_ylim(0, 1.6)
for d, lab, dx in [("2009-03", "Mar 2009 (market bottom)", -1600), ("2020-04", "Apr 2020 (rebound)", -1500)]:
    x = pd.Period(d, "M").to_timestamp()
    ax[1].annotate(f"{lab}: exposure {w.loc[d]:.2f}", (x, w.loc[d]), (x + pd.Timedelta(days=dx), 1.25), ha="left",
                   fontsize=8, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.9),
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=.7))
for g, col, lab in [(gu, GREY, "Buy and hold"), (gm, BLUE, "Timed")]:
    dd = 100 * (g / g.cummax() - 1); ax[2].plot(t, dd, color=col, lw=1.4, label=f"{lab}: worst {dd.min():.0f}%")
ax[2].set_ylabel("Drawdown (%)"); ax[2].legend(loc="lower right", ncol=2)
fig.text(0, -0.01, NOTE + " Source: results/main/nets_1_us_market.csv.", fontsize=7.5, color=INK2)
save(fig, "fig1_us_market.png")

# ---------------------------------------------------------------- Fig 2: MT x VT
T = pd.read_csv(R / "1_us_market.csv"); T = T[T.period == config.MAIN].set_index("rule")
rules = [("MM on RV", "Last month's realised variance"), ("MM on GARCH", "GARCH(1,1) forecast"),
         ("MM on GJR", "GJR-GARCH forecast"), ("MM on VIX^2", "VIX$^2$ (option-implied)")]
fig, ax = plt.subplots(figsize=(9, 3.9))
y = np.arange(len(rules))[::-1]
mt, vt = T.loc[[r for r, _ in rules], "MT"].to_numpy(), T.loc[[r for r, _ in rules], "VT"].to_numpy()
ax.barh(y + .19, vt, .36, color=BLUE, label="VT: variance timing (risk side)")
ax.barh(y - .19, mt, .36, color=ORANGE, label="MT: mean timing (return side)")
ax.axvline(1, color=INK, lw=1)
for i, (r, lab) in enumerate(rules):
    ax.text(vt[i] + .01, y[i] + .19, f"{vt[i]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.text(mt[i] + .01, y[i] - .19, f"{mt[i]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.text(1.40, y[i], f"MT x VT = {mt[i] * vt[i]:.2f}   net: {T.loc[r, 'Sharpe'] / T.loc['Unmanaged', 'Sharpe']:.2f}",
            va="center", fontsize=8.5, color=INK)
ax.set_yticks(y, [lab for _, lab in rules]); ax.set_xlim(.8, 1.68); ax.set_xticks([.8, .9, 1, 1.1, 1.2, 1.3])
ax.set_xlabel("Ratio (1 = no gain). Sharpe(timed) / Sharpe(buy and hold) = MT x VT before costs")
ax.legend(loc="lower center", bbox_to_anchor=(.4, -.36), ncol=2)
title(ax, "Better risk forecasts gain variance timing but lose mean timing\n(US market, 1998-2025)",
             loc="left", color=INK, fontsize=11.5)
fig.text(0, -.2, "MT = E[w R] / E[R], VT = sd(R) / sd(w R), exposure w scaled to mean 1, before costs; 'net' = after 10 bp costs. "
         "VIX$^2$ data start in 1990. "
         "Source: results/main/1_us_market.csv.", fontsize=7.5, color=INK2)
save(fig, "fig2_mt_vt.png")


# ---------------------------------------------------------------- Fig 3: gains before and after 2016
def jkp_pooled(mix):
    txt = (R / "6_intl_jkp.txt").read_text(encoding="utf-8")
    block = txt.split(f"================ {mix}\n")[1].split("================")[0]
    out = {}
    for p in ["1998-2015", "2016-2025"]:
        sec = block.split(f"--- {p}")[1]
        out[p] = parse_ci(re.search(r"pooled 7 regions, timing gain: (.*)", sec).group(1))
    return out


def us_row(csv, filt, col):
    D = pd.read_csv(R / csv); out = {}
    for p in ["1998-2015", "2016-2025"]:
        out[p] = parse_ci(D[filt(D) & (D.period == p)][col].iloc[0])
    return out


F7 = pd.read_csv(R / "7_intl_french.csv")
fr = lambda mix: {p: parse_ci(F7[(F7.mix == mix) & (F7.period == p) & (F7.version == "Whole mix")]["pooled gain"].iloc[0])
                  for p in ["1998-2015", "2016-2025"]}
rows = [("US market", us_row("1_us_market.csv", lambda D: D.rule == "MM on GARCH", "dSR vs unmanaged [95% CI]")),
        ("US mix Mkt+RMW+CMA+Mom*", us_row("5_us_main_result.csv", lambda D: (D.mix == "Mkt + RMW + CMA + Mom") &
                                           (D.version == "MM-GARCH") & (D.view == "as is"), "timing gain [95% CI]")),
        ("JKP 7 regions, Mkt+Prof+Inv+Mom", jkp_pooled("Mkt + Prof + Inv + Mom")),
        ("French 3 regions, RMW+CMA+Mom", fr("RMW + CMA + Mom")),
        ("JKP 7 regions, momentum", jkp_pooled("Mom alone")),
        ("French 3 regions, momentum", fr("Mom alone"))]
fig, ax = plt.subplots(figsize=(9, 4.4))
for i, (lab, d) in enumerate(rows[::-1]):
    for p, col, off in [("1998-2015", BLUE, .15), ("2016-2025", ORANGE, -.15)]:
        g, lo, hi = d[p]
        ax.plot([lo, hi], [i + off] * 2, color=col, lw=2.2, solid_capstyle="round")
        ax.plot(g, i + off, "o", ms=7, color=col, mec="white", mew=1.2,
                label={"1998-2015": "1998-2015", "2016-2025": "2016-2025"}[p] if i == 0 else None)
        ax.text(hi + .02, i + off, f"{g:+.2f}", va="center", fontsize=8, color=INK2)
ax.axvline(0, color=INK, lw=1)
ax.set_yticks(range(len(rows)), [lab for lab, _ in rows[::-1]])
ax.set_xlabel("Timing gain in annualised Sharpe ratio (timed minus buy and hold), 95% block-bootstrap interval")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2); ax.set_xlim(-.45, .85)
title(ax, "The gain was there before 2016 and mostly gone after; momentum holds up best", loc="left", color=INK,
             fontsize=12, pad=24)
fig.text(0, -.04, "* chosen with hindsight (#1 of 63 mixes in 1998-2025, #14 before 1998). Pooled = equal-weight average "
         "of the regions. " + NOTE, fontsize=7.5, color=INK2, wrap=True)
save(fig, "fig3_gains_by_period.png")

# ---------------------------------------------------------------- Fig 4: multiple testing
M = pd.read_csv(R / "8_multiple_testing.csv"); n = len(M)
steps = [("All timing comparisons", n), ("Significant at 5%, no correction", int((M.p_raw < .05).sum())),
         ("Expected by luck alone (5%)", round(.05 * n)), ("Benjamini-Hochberg (false discovery rate)", int((M.p_bh < .05).sum())),
         ("Holm (family-wise)", int((M.p_holm < .05).sum())), ("Romano-Wolf (family-wise, joint bootstrap)",
                                                               int((M.p_romano_wolf < .05).sum()))]
fig, ax = plt.subplots(figsize=(9, 3.6))
y = np.arange(len(steps))[::-1]
cols = [GREY, BLUE, "#c9c8c3", BLUE, BLUE, BLUE]
ax.barh(y, [v for _, v in steps], color=cols, height=.6)
for yi, (lab, v) in zip(y, steps):
    ax.text(v + 3, yi, str(v), va="center", fontsize=9, color=INK)
ax.set_yticks(y, [lab for lab, _ in steps]); ax.set_xlim(0, n * 1.12); ax.grid(axis="y", visible=False)
top = M.sort_values("p_romano_wolf").iloc[0]
title(ax, "After correcting for everything we tried, one comparison survives", loc="left", color=INK, fontsize=12)
ax.set_xlabel("Number of comparisons (timed vs buy and hold, 1998-2025)")
nice = top.group.replace("JKP fra", "JKP France").replace("JKP ", "JKP data, ")
fig.text(0, -.06, f"Survivor: {nice}, {top.series.replace(' alone', '')}, {top.rule}: Sharpe gain {top.dSR:+.2f}, t = {top.t:.1f}, "
         f"Romano-Wolf p = {top.p_romano_wolf:.3f}. Source: results/main/8_multiple_testing.csv.", fontsize=7.5, color=INK2)
save(fig, "fig4_multiple_testing.png")

# ---------------------------------------------------------------- Fig 5: recovery state, the mechanism
r = data.us_daily(["Mkt-RF"])["Mkt-RF"]
rv1 = data.us_rv(["Mkt-RF"])["Mkt-RF"].shift(1).reindex(e.index)
rv5, rv63 = rv_states(r, e.index)
w_rv = mm_weights(rv1, e, S)[0]
q80 = rv1.expanding(120).quantile(0.8)
high = rv1 > q80
state = pd.Series(np.where(high & (rv5 < rv63), "Recovery", np.where(high, "Panic", "Normal")), index=e.index)
state[q80.isna()] = None
win = ("1975-01", "2025-12")
D = pd.DataFrame({"state": state, "ret": e, "var": e ** 2, "w": w_rv}).loc[win[0]:win[1]].dropna()
order = ["Normal", "Panic", "Recovery"]
G = D.groupby("state").agg(ret=("ret", "mean"), sd=("ret", "std"), n=("ret", "size"), w=("w", "mean")).loc[order]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
col = [GREY, RED, BLUE]
ax[0].bar(order, 100 * G.ret, color=col, width=.6, yerr=196 * G.sd / np.sqrt(G.n), capsize=4,
          error_kw=dict(ecolor=INK2, lw=1))
for i, s in enumerate(order):
    ax[0].text(i, 0.08 if G.ret[s] < 0 else -0.35, f"n={G.n[s]}", ha="center", fontsize=8, color=INK2)
ax[0].axhline(0, color=INK, lw=.8)
ax[0].set_ylabel("Next-month excess return (% per month)"); ax[0].set_title("What the month pays", loc="left", color=INK)
ax[1].bar(order, G.w, color=col, width=.6); ax[1].axhline(1, color=GREY, ls="--", lw=1)
for i, s in enumerate(order):
    ax[1].text(i, G.w[s] + .03, f"{G.w[s]:.2f}", ha="center", fontsize=9, color=INK)
ax[1].set_ylabel("Average volatility-timed exposure"); ax[1].set_title("How much timing holds", loc="left", color=INK)
title(fig, f"US market {win[0][:4]}-{win[1][:4]}: timing is most out of the market in recovery months",
             x=0.0, ha="left", color=INK, fontsize=12, y=1.04)
fig.text(0, -.1, "Panic = last month's realised variance in its top 20% (expanding, real time) and still rising "
         "(5-day > 63-day variance); Recovery = top 20% but falling. Bars: mean with 95% interval. "
         "Exposure: MM on realised variance.", fontsize=7.5, color=INK2, wrap=True)
fig.tight_layout()
save(fig, "fig5_recovery_state.png")
print(G.assign(ret_pct=100 * G.ret).round(3).to_string())
