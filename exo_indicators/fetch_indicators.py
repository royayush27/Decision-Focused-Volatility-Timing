"""Rebuild exo_indicators/indicators_monthly.csv from the original sources.

Sources (all public, no key):
  FRED: BAA, AAA, GS10, TB3MS (monthly averages), USEPUINDXD (daily EPU), EMVOVERALLEMV (monthly EMV),
        EPUTRADE, EPUMONETARY, EPUNATSEC (categorical EPU),
        DBAA, DAAA (daily, for end-of-month spread)
  San Francisco Fed: Daily News Sentiment Index (daily, from 1980)
Columns (month = calendar month the value belongs to; the test scripts apply the lags):
  DEF = BAA - AAA, CS = BAA - GS10, TERM = GS10 - TB3MS, EPU = mean of daily EPU in the month,
  EMV, DEF_eom = DBAA - DAAA on the last trading day, NS_eom = last daily sentiment value, NS_avg = monthly mean,
  EPU_TRADE / EPU_MON / EPU_NATSEC = categorical EPU (trade, monetary, national security), monthly.

Not run in the sandbox (it cannot reach FRED). The committed CSV was built from the same sources through a
browser and checksum-verified. On first run this script compares its output with that file and prints the
largest difference per column. Small differences can appear if a source revised its history.
Usage (from the project root):  python exo_indicators/fetch_indicators.py  [--write]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = Path("exo_indicators/indicators_monthly.csv")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SFFED = "https://www.frbsf.org/wp-content/uploads/news-sentiment-chart-1.csv"


def fred(sid):
    d = pd.read_csv(FRED.format(sid), na_values=[".", ""])
    d.columns = ["date", "v"]
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna().set_index("date")["v"]


def monthly_key(s):
    return s.index.strftime("%Y-%m")


def main():
    mon = {}
    for sid in ("BAA", "AAA", "GS10", "TB3MS", "EMVOVERALLEMV", "EPUTRADE", "EPUMONETARY", "EPUNATSEC"):
        s = fred(sid)
        mon[sid] = pd.Series(s.to_numpy(), index=monthly_key(s))
    epu = fred("USEPUINDXD")
    epu_m = epu.groupby(monthly_key(epu)).mean()
    last = lambda s: s.groupby(monthly_key(s)).last()
    def_eom = last(fred("DBAA")) - last(fred("DAAA"))

    ns = pd.read_csv(SFFED)
    ns.columns = ["date", "v"]
    ns["date"] = pd.to_datetime(ns["date"], format="%m/%d/%Y")
    ns = ns.dropna().set_index("date").sort_index()["v"]
    ns_eom, ns_avg = last(ns), ns.groupby(monthly_key(ns)).mean()

    months = pd.period_range("1965-01", "2025-12", freq="M").astype(str)
    df = pd.DataFrame(index=pd.Index(months, name="month"))
    df["DEF"] = (mon["BAA"] - mon["AAA"]).reindex(months).round(2)
    df["CS"] = (mon["BAA"] - mon["GS10"]).reindex(months).round(2)
    df["TERM"] = (mon["GS10"] - mon["TB3MS"]).reindex(months).round(2)
    df["EPU"] = epu_m.reindex(months).round(1)
    df["EMV"] = mon["EMVOVERALLEMV"].reindex(months).round(2)
    df["DEF_eom"] = def_eom.reindex(months).round(2)
    df["NS_eom"] = ns_eom.reindex(months).round(2)
    df["NS_avg"] = ns_avg.reindex(months).round(3)
    df["EPU_TRADE"] = mon["EPUTRADE"].reindex(months).round(1)      # categorical EPU, monthly, published with a lag
    df["EPU_MON"] = mon["EPUMONETARY"].reindex(months).round(1)
    df["EPU_NATSEC"] = mon["EPUNATSEC"].reindex(months).round(1)

    if OUT.exists():
        old = pd.read_csv(OUT, index_col="month")
        print("largest absolute difference vs committed file (rounding alone gives at most one unit in the last decimal):")
        for c in old:
            print(f"  {c:8s} {np.nanmax(np.abs(df[c].to_numpy() - old[c].reindex(df.index).to_numpy())):.4f}")
    if "--write" in sys.argv or not OUT.exists():
        df.to_csv(OUT)
        print("wrote", OUT)


if __name__ == "__main__":
    main()
