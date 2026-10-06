"""Data loaders. Every monthly series is an EXCESS return in decimals, indexed 'YYYY-MM'.

US (Ken French library, data/): 6 factors Mkt-RF, SMB, HML, RMW, CMA, Mom (already excess / long-short) and 10 industry
portfolios (total returns: the risk-free rate is subtracted here), monthly and daily, July 1963 on.
International: JKP Global Factor Data (data/jkp/, vw_cap, USD) and Ken French regional factors (data/ff_intl/, USD).
"""
from functools import lru_cache

import pandas as pd

from config import DATA

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
INDUSTRIES = ["NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other"]
SERIES = FACTORS + INDUSTRIES


@lru_cache(None)
def rf():
    return pd.read_csv(DATA / "ff_monthly_rf.csv", index_col="month")["RF"]


@lru_cache(None)
def _us_monthly():
    return pd.read_csv(DATA / "ff_monthly_baselinedata.csv")


@lru_cache(None)
def _us_daily():
    d = pd.read_csv(DATA / "ff_daily_garchdata_long.csv")
    d["date"] = pd.to_datetime(d["Date"], format="%d-%m-%Y")
    return d


def us_monthly(series=SERIES, end="2025-12"):
    """Monthly excess returns, July 1963 - end."""
    m = _us_monthly()
    X = pd.DataFrame({s: m[m["series"] == s].set_index("month").sort_index()["ret_m"] for s in series}).loc["1963-07":end]
    for s in series:
        if s not in FACTORS:
            X[s] = X[s] - rf().reindex(X.index)
    return X


def us_rv(series=SERIES, end="2025-12"):
    """Realised variance of each month (sum of squared daily returns), from the monthly file."""
    m = _us_monthly()
    return pd.DataFrame({s: m[m["series"] == s].set_index("month").sort_index()["rv"] for s in series}).loc["1963-07":end]


def us_daily(series=SERIES):
    """Daily returns (industries: total returns, as used for their GARCH and realised variance)."""
    d = _us_daily()
    return pd.DataFrame({s: d[d["series"] == s].set_index("date")["ret"] for s in series}).sort_index()


def vix_sq_monthly():
    """VIX^2 at the end of each month as a monthly variance (decimal)."""
    vx = pd.read_csv(DATA / "VIXCLS.csv"); vx.columns = ["date", "vix"]
    vx["date"] = pd.to_datetime(vx["date"]); vx["vix"] = pd.to_numeric(vx["vix"], errors="coerce")
    v = vx.dropna().set_index("date")["vix"].resample("ME").last(); v.index = v.index.strftime("%Y-%m")
    return v ** 2 / 12 / 1e4


def jkp(region):
    """JKP factors: mkt, ope_be (~RMW), at_gr1 (~CMA), ret_12_1 (~Mom). Returns (daily, monthly)."""
    def ld(freq):
        x = pd.read_csv(DATA / "jkp" / f"{region}_{freq}.csv", dtype={"date": str})
        x.index = pd.to_datetime(x.pop("date"), format="%Y%m%d")
        return x
    D, M = ld("daily"), ld("monthly")
    M.index = M.index.to_period("M").astype(str)
    return D, M


def ff_intl(region):
    """Ken French regional 5 factors + momentum. Returns (daily, monthly) with columns as FACTORS."""
    D = pd.read_csv(DATA / "ff_intl" / f"{region}_daily.csv", dtype={"date": str})
    D.index = pd.to_datetime(D.pop("date"), format="%Y%m%d")
    M = pd.read_csv(DATA / "ff_intl" / f"{region}_monthly.csv", dtype={"date": str})
    M.index = M.pop("date").map(lambda s: f"{s[:4]}-{s[4:6]}")
    return D[FACTORS], M[FACTORS]
