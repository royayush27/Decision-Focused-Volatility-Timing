"""Rule variants built on top of the Moreira-Muir exposure (all real time, information at the end of month t-1).
  Xu switches     MM on GARCH; cash if the past 12-month mean return < 0; exposure 1 if the 60-month regression of
                  returns on last month's RV has intercept <= 0 (Xu 2024).
  Conditional     MM on GARCH only when last month's RV is in its top or bottom 20% (expanding quantiles), else 1
                  (Bongaerts, Kang and van Dijk 2020).
  Recovery state  MM on RV, but at least 1 when last month's RV is in its top 20% and 5-day RV < 63-day RV (vol has peaked).
The cap of the spec is applied again at the end."""
import numpy as np
import pandas as pd

from core.rule import cap_weights


def rolling_intercept(r, v, window=60):
    a = pd.Series(np.nan, index=r.index)
    for i in range(window, len(r)):
        y, x = r.iloc[i - window:i].to_numpy(), v.iloc[i - window:i].to_numpy(); ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 48:
            a.iloc[i] = np.polyfit(x[ok], y[ok], 1)[1]
    return a


def rv_states(r_daily, months):
    """5-day and 63-day realised variance (scaled to a month) at the end of t-1, for each month t."""
    eom = r_daily.groupby(r_daily.index.to_period("M")).apply(lambda z: z.index[-1])
    f = lambda k: pd.Series(((r_daily ** 2).rolling(k).sum() * 21 / k).loc[eom.values].to_numpy(),
                            index=eom.index.astype(str)).shift(1).reindex(months)
    return f(5), f(63)


def xu(w_garch, e, rv1, spec):
    hold = e.rolling(12).mean().shift(1) > 0
    scale = rolling_intercept(e, rv1) > 0
    return cap_weights(w_garch.where(scale, 1.0).where(hold, 0.0).where(w_garch.notna()), spec)


def conditional(w_garch, rv1, spec):
    q20, q80 = rv1.expanding(120).quantile(0.2), rv1.expanding(120).quantile(0.8)
    return cap_weights(w_garch.where((rv1 >= q80) | (rv1 <= q20), 1.0).where(w_garch.notna()), spec)


def recovery(w_rv, rv1, rv5, rv63, spec):
    q80 = rv1.expanding(120).quantile(0.8)
    rec = (rv1 > q80) & (rv5 < rv63)
    return cap_weights(w_rv.where(~rec, np.maximum(w_rv, 1.0)).where(w_rv.notna()), spec)
