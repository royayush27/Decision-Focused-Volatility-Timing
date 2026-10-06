"""Regime-dependent factor allocation (Ayush's GitHub project) rebuilt on French factors, real time vs look-ahead.
Offensive basket (his defaults, BAB not in French data, weights renormalised): MKT .43, Mom .43, SMB .14.
Defensive basket: RMW .5, CMA .5. Regime: VIX > 20 -> defensive.
  'same-month VIX' = VIX at the END of the month being traded (what the GitHub script does: look-ahead)
  'lagged VIX'     = VIX at the end of the previous month (available when the weights are set)
Then vol-managed versions: each factor scaled by c/v_hat (QMLE GARCH, c frozen on 1970-1999), same baskets.
Sample 1990-02 to 2024-12 (VIX era). Gross returns, plus net at 5 bps per unit of factor-weight turnover."""
import os, sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "dfl_diagnosis", _P(__file__).resolve().parent):
    sys.path.insert(0, str(_p))
os.chdir(_ROOT)   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd

from calibrate_c import exposure, solve_c

F = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
m = pd.read_csv("data/ff_monthly_baselinedata.csv")
R = pd.DataFrame({f: m[m.series == f].set_index("month").sort_index()["ret_m"] for f in F}).loc["1970-01":"2025-12"]
v = pd.read_csv("data/VIXCLS.csv"); v.columns = ["date", "vix"]
v["date"] = pd.to_datetime(v["date"]); v = v.dropna()
vix = v.set_index("date")["vix"].resample("ME").last(); vix.index = vix.index.strftime("%Y-%m")
vix = vix.reindex(R.index)
OFF = {"Mkt-RF": .3 / .7, "Mom": .3 / .7, "SMB": .1 / .7}
DEF = {"RMW": .5, "CMA": .5}
# vol-managed factor returns (each factor scaled to mean exposure 1 on 1970-1999)
VM = pd.DataFrame(index=R.index)
for f in F:
    vh = pd.read_csv(f"dfl_diagnosis/panel/{f}_garch.csv", index_col="month")["qmle"].reindex(R.index)
    VM[f] = exposure(vh, solve_c(vh.loc[:"1999-12"])) * R[f]


def basket(ret, stress):
    off = sum(w * ret[f] for f, w in OFF.items()); de = sum(w * ret[f] for f, w in DEF.items())
    return off.where(~stress, de)


def stats(z, w_turn=None):
    z = z.loc["1990-02":"2024-12"].dropna()
    wl = (1 + z).cumprod()
    return {"ann_ret": z.mean() * 12, "vol": z.std() * np.sqrt(12), "SR": z.mean() / z.std() * np.sqrt(12),
            "MaxDD": (wl / wl.cummax() - 1).min()}


hi_same, hi_lag = vix > 20, vix.shift(1) > 20
S = {
    "EW 6 factors (static)": R[F].mean(axis=1),
    "VIX rule, same-month VIX (GitHub)": basket(R, hi_same),
    "VIX rule, lagged VIX (real time)": basket(R, hi_lag),
    "offensive basket always": basket(R, pd.Series(False, index=R.index)),
    "defensive basket always": basket(R, pd.Series(True, index=R.index)),
    "EW 6 vol-managed factors": VM[F].mean(axis=1),
    "VIX rule lagged, vol-managed factors": basket(VM, hi_lag),
}
T = pd.DataFrame({k: stats(z) for k, z in S.items()}).T
print("1990-2024, gross monthly returns")
print(T.round(3).to_string())
print("\nmonths in stress regime (VIX>20):", int(hi_lag.loc["1990-02":"2024-12"].sum()), "of",
      int(hi_lag.loc["1990-02":"2024-12"].notna().sum()))
# why the look-ahead matters: correlation of same-month VIX change with the market return
dv = vix.diff()
print("corr(month-t VIX change, month-t market return):", round(dv.corr(R["Mkt-RF"]), 2),
      "| corr(month t-1 VIX change, month-t market return):", round(dv.shift(1).corr(R["Mkt-RF"]), 2))

# paired bootstrap: lagged VIX rule vs static EW and vs offensive-always
rng = np.random.default_rng(0)
def boot(a, b, BB=4000, BL=6):
    a, b = a.loc["1990-02":"2024-12"].to_numpy(), b.loc["1990-02":"2024-12"].to_numpy(); n = len(a)
    st = rng.integers(0, n, (BB, int(np.ceil(n / BL)))); I = ((st[:, :, None] + np.arange(BL)) % n).reshape(BB, -1)[:, :n]
    sr = lambda M: M.mean(1) / M.std(1, ddof=1) * np.sqrt(12)
    d = sr(a[I]) - sr(b[I]); d0 = (a.mean() / a.std(ddof=1) - b.mean() / b.std(ddof=1)) * np.sqrt(12)
    return f"{d0:+.3f} [{np.percentile(d-d.mean()+d0,2.5):+.3f}, {np.percentile(d-d.mean()+d0,97.5):+.3f}]"
print("\ndSR (95% block-bootstrap CI):")
for a, b in (("VIX rule, lagged VIX (real time)", "EW 6 factors (static)"),
             ("VIX rule, lagged VIX (real time)", "offensive basket always"),
             ("EW 6 vol-managed factors", "EW 6 factors (static)"),
             ("VIX rule lagged, vol-managed factors", "EW 6 vol-managed factors")):
    print(f"  {a} - {b}: {boot(S[a], S[b])}")
