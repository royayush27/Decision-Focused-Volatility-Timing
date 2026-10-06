"""Step 1: in-sample replication of Moreira & Muir (2017) Table I, Panel A (full-sample c, RV weights)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data import load, FACTORS


def realized_variance(daily):
    """MM eq. (2): sum over the month of squared deviations of daily returns from the monthly mean."""
    g = daily.groupby(daily.index.to_period("M"))
    return g.apply(lambda x: ((x - x.mean()) ** 2).sum(min_count=1))


def mm_table(end="2015-12", start=None):
    m, d = load(False), load(True)
    rv = realized_variance(d[FACTORS])
    rows = {}
    for f in FACTORS:
        df = pd.DataFrame({"f": m[f], "rv_lag": rv[f].shift(1)}).dropna()
        df = df.loc[start:end] if start else df.loc[:end]
        raw = df["f"] / df["rv_lag"]
        c = df["f"].std() / raw.std()          # same unconditional sd as buy-and-hold (full sample, as in MM)
        fs = c * raw
        X = sm.add_constant(df["f"])
        r = sm.OLS(fs, X).fit(cov_type="HC0")
        rmse = np.sqrt(r.mse_resid)
        rows[f] = {
            "beta": r.params["f"], "alpha_%pa": 12 * 100 * r.params["const"],
            "se_alpha": 12 * 100 * r.bse["const"], "N": int(r.nobs), "R2": r.rsquared,
            "RMSE": 12 * 100 * rmse, "appraisal_ann": np.sqrt(12) * r.params["const"] / rmse,
            "SR_old": np.sqrt(12) * df["f"].mean() / df["f"].std(),
            "SR_new": np.sqrt(12) * fs.mean() / fs.std(),
            "start": str(df.index[0]),
        }
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t = mm_table()
    print(t.round(3))
    print("\nMM Table I reported alphas: Mkt 4.86 (1.56), SMB -0.58, HML 1.97, Mom 12.51, RMW 2.44, CMA 0.38")
    print("MM reported betas: 0.61 0.62 0.57 0.47 0.62 0.68 ; R2: .37 .38 .32 .22 .38 .46")
    t.to_csv("results/mm_replication_1926_2015.csv")
    t2 = mm_table(end="2026-08")
    print("\nFull sample to 2026-08:\n", t2.round(3))
    t2.to_csv("results/mm_replication_to_2026.csv")
