import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))   # project root, for backtest.py and calibrate_c.py

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from backtest import simulate
from calibrate_c import exposure, solve_c

# Step A: decision-focused training of GARCH(1,1) and GJR-GARCH.
# theta = (omega, alpha, gamma, beta); gamma = 0 for plain GARCH. r is in percent.
# h_t = omega + (alpha + gamma*1[r_{t-1}<0]) r_{t-1}^2 + beta h_{t-1}
# The decision objective is the net utility of w = clip(c / v_hat, 0, 2) with c re-solved so that
# mean exposure is 1. That makes utility invariant to scaling v_hat, so omega is fixed at its QMLE
# value and only the remaining parameters are optimised. The level of v_hat is then re-anchored to the
# mean realised variance of the training months, so forecasts are comparable across yearly refits.

H = 21
KAPPA = 5e-4
GAMMA = 3.0
W_MAX = 2.0
FIRST_REFIT, LAST_YEAR = 2000, 2025


def load(daily_csv, monthly_csv, rf_csv):
    d = pd.read_csv(daily_csv)
    m = d[d["series"] == "Mkt-RF"].copy()
    m["Date"] = pd.to_datetime(m["Date"], format="%d-%m-%Y")
    r = m.set_index("Date")["ret"].sort_index() * 100
    base = pd.read_csv(monthly_csv, index_col="month")
    ex = base["ret_m"]
    rf = pd.read_csv(rf_csv, index_col="month")["RF"].reindex(ex.index)
    assert rf.notna().all()
    months = pd.PeriodIndex(ex.index, freq="M")
    origin = np.searchsorted(r.index.values, months.to_timestamp().values) - 1  # last trading day before the month
    return r.to_numpy(), r.index, ex, rf, months, origin, base["rv"]


def persistence(th):
    return th[1] + th[2] / 2 + th[3]


def valid(th):
    om, a, g, b = th
    return om > 0 and a >= 0 and b >= 0 and a + g >= 0 and persistence(th) < 0.999


def filter_h(th, r, h0, n):
    # h[0] = h0, h[t] uses r[t-1]; recursion h_t - beta h_{t-1} = omega + shock term, done with lfilter
    om, a, g, b = th
    rr = r[: n - 1]
    u = om + (a + g * (rr < 0)) * rr ** 2
    y, _ = lfilter([1.0], [1.0, -b], u, zi=[b * h0])
    return np.r_[h0, y]


def monthly_vhat(th, r, h0, origin):
    # 21-day variance forecast at each origin: day 1 exact, then h_k = omega + rho h_{k-1}
    om, a, g, b = th
    h = filter_h(th, r, h0, origin.max() + 1)
    rT = r[origin]
    p0 = om + (a + g * (rT < 0)) * rT ** 2 + b * h[origin]
    rho = persistence(th)
    hss = om / (1 - rho)
    return (H * hss + (p0 - hss) * (1 - rho ** H) / (1 - rho)) / 1e4


def nll(th, r, h0):
    if not valid(th):
        return 1e10
    h = filter_h(th, r, h0, len(r))
    if (h <= 0).any() or not np.isfinite(h).all():
        return 1e10
    return 0.5 * np.mean(np.log(h[1:]) + r[1:] ** 2 / h[1:])


def qmle(r_tr, gjr, h0):
    best = None
    for rho in (0.97, 0.90):
        th0 = np.array([0.02, 0.06, 0.04 if gjr else 0.0, rho - 0.06 - (0.02 if gjr else 0.0)])
        free = [0, 1, 2, 3] if gjr else [0, 1, 3]

        def f(p):
            th = th0.copy()
            th[free] = p
            return nll(th, r_tr, h0)

        res = minimize(f, th0[free], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 4000})
        th = th0.copy()
        th[free] = res.x
        if best is None or res.fun < best[1]:
            best = (th, res.fun)
    return best[0]


class Train:
    # everything the utility needs for one training window
    def __init__(self, r, h0, origin, ex, rf, months, upto):
        tm = months < pd.Period(f"{upto}-01")
        self.r, self.h0 = r, h0
        self.origin = origin[tm]
        self.ex, self.rf = ex[tm], rf[tm]
        self.idx = ex.index[tm]

    def utility(self, th):
        v = monthly_vhat(th, self.r, self.h0, self.origin)
        if not (np.isfinite(v).all() and (v > 0).all()):
            return -1e10
        c = solve_c(v, W_MAX)                                  # c re-solved: exposure level is fixed
        w = pd.Series(exposure(v, c, W_MAX), index=self.idx)
        s = simulate(w, self.ex, self.rf, KAPPA)
        mu, var = s.net_ex.mean() * 12, s.net_ex.var(ddof=1) * 12
        return mu - GAMMA / 2 * var


def fit_decision(q, gjr, tr):
    # optimise (alpha, [gamma], beta) with omega held at the QMLE value, Nelder-Mead on the exact objective
    free = [1, 2, 3] if gjr else [1, 3]

    def neg(p):
        th = q.copy()
        th[free] = p
        return 1e10 if not valid(th) else -tr.utility(th)

    res = minimize(neg, q[free], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 800})
    res = minimize(neg, res.x, method="Nelder-Mead",       # one restart to escape a collapsed simplex
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 800})
    th = q.copy()
    th[free] = res.x
    return th, -res.fun, res.nfev


def run(model, data, out):
    gjr = model == "gjr"
    r, dates, ex, rf, months, origin, rv = data
    vhat = pd.Series(np.nan, index=ex.index)
    rows = []
    for y in range(FIRST_REFIT, LAST_YEAR + 1):
        n_tr = np.searchsorted(dates.values, np.datetime64(f"{y}-01-01"))
        r_tr = r[:n_tr]
        h0 = r_tr.var()
        tr = Train(r, h0, origin, ex, rf, months, y)
        q = qmle(r_tr, gjr, h0)
        th, u_dec, nfev = fit_decision(q, gjr, tr)
        # utility ignores the level of v_hat, so anchor it: mean forecast = mean realised variance on the
        # training months. This keeps the level comparable across yearly refits (the frozen c needs that).
        k = rv[tr.idx].mean() / monthly_vhat(th, r, h0, tr.origin).mean()
        in_year = (months >= pd.Period(f"{y}-01")) & (months <= pd.Period(f"{y}-12"))
        vhat[in_year] = k * monthly_vhat(th, r, h0, origin[in_year])
        if y == FIRST_REFIT:                                # calibration-window forecasts use the first fit
            before = months < pd.Period(f"{y}-01")
            vhat[before] = k * monthly_vhat(th, r, h0, origin[before])
        unmanaged = simulate(pd.Series(1.0, index=tr.idx), tr.ex, tr.rf, KAPPA).net_ex
        rows.append({"year": y, "omega": q[0], "a_qmle": q[1], "g_qmle": q[2], "b_qmle": q[3],
                     "a_dec": th[1], "g_dec": th[2], "b_dec": th[3], "rho_qmle": persistence(q),
                     "rho_dec": persistence(th), "util_qmle": tr.utility(q), "util_dec": u_dec,
                     "util_unmanaged": unmanaged.mean() * 12 - GAMMA / 2 * unmanaged.var(ddof=1) * 12,
                     "level_k": k, "nfev": nfev, "n_train_months": len(tr.idx)})
        print(f"{model} {y}: rho {rows[-1]['rho_qmle']:.3f}->{rows[-1]['rho_dec']:.3f}  "
              f"U(train) qmle {rows[-1]['util_qmle']:.4f}  dec {u_dec:.4f}  unmanaged {rows[-1]['util_unmanaged']:.4f}",
              flush=True)
    assert vhat.notna().all() and (vhat > 0).all()
    out.mkdir(parents=True, exist_ok=True)
    vhat.rename("v_hat").to_frame().to_csv(out / f"{model}_dec.csv", index_label="month")
    pd.DataFrame(rows).to_csv(out / f"{model}_dec_diag.csv", index=False)
    print("saved", out / f"{model}_dec.csv", len(vhat), "rows")


def selftest(data, out):
    r, dates, ex, rf, months, origin, rv = data
    n_tr = np.searchsorted(dates.values, np.datetime64("2000-01-01"))
    r_tr = r[:n_tr]
    h0 = r_tr.var()
    i = list(ex.index).index("2000-01")

    # 1. closed-form 21-day sum equals a brute-force 21-step loop (GJR parameters, so the sign term is tested)
    th = np.array([0.0068, 0.03, 0.08, 0.90])
    T = origin[i]
    h = filter_h(th, r, h0, T + 1)
    p = np.empty(H)
    p[0] = th[0] + (th[1] + th[2] * (r[T] < 0)) * r[T] ** 2 + th[3] * h[T]
    for k in range(1, H):
        p[k] = th[0] + persistence(th) * p[k - 1]
    a, b = monthly_vhat(th, r, h0, origin[[i]])[0], p.sum() / 1e4
    print(f"1. closed form vs loop: {a:.12f} {b:.12f}  ok={np.isclose(a, b, rtol=1e-10)}")

    # 2. scale invariance of the decision utility
    tr = Train(r, h0, origin, ex, rf, months, 2000)
    q = qmle(r_tr, False, h0)
    u = [tr.utility(np.array([q[0] * k, q[1] * k, 0.0, q[3]])) for k in (1, 2, 0.5)]
    print("2. utility at (k*omega, k*alpha, beta), k=1,2,0.5:", np.round(u, 5), " spread", round(max(u) - min(u), 5))

    # 3. match to the statistical forecast file for 2000-01
    f = out / "garch_stat.csv"
    mine = monthly_vhat(q, r, h0, origin[[i]])[0]
    print(f"3. QMLE theta {np.round(q, 5)}  v_hat 2000-01 = {mine:.6f}", end="")
    if f.exists():
        ref = pd.read_csv(f, index_col="month")["v_hat"]["2000-01"]
        print(f"  garch_stat.csv = {ref:.6f}  ratio {mine / ref:.4f}")
    else:
        print()


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["garch", "gjr", "both"], default="both")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--daily", type=Path, default=root / "data/ff_daily_garchdata_long.csv")
    p.add_argument("--monthly", type=Path, default=root / "results_1970" / "market_forecasts.csv")
    p.add_argument("--rf", type=Path, default=root / "data/ff_monthly_rf.csv")
    p.add_argument("--out", type=Path, default=root / "results_1970" / "forecasts")
    a = p.parse_args()
    data = load(a.daily, a.monthly, a.rf)
    if a.selftest:
        selftest(data, a.out)
        return
    for m in (["garch", "gjr"] if a.model == "both" else [a.model]):
        run(m, data, a.out)


if __name__ == "__main__":
    main()
