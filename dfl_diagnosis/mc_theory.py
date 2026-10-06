import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from calibrate_c import exposure, solve_c
from decision_garch import monthly_vhat, persistence, qmle, valid

# Monte Carlo: do the three losses target the same parameter, and which gives better out-of-sample decisions?
# Days per month fixed at 21. Train 30 years, test 30 years, same draw. Gaussian innovations.
D, TRAIN_M, TEST_M = 21, 360, 360
GAMMA, W_MAX = 3.0, 2.0


def sim(case, n_months, rng):
    n = n_months * D + 1
    om, a, g, b = case["theta"]
    z = rng.standard_normal(n)
    r = np.empty(n); h = np.empty(n)
    h[0] = om / (1 - a - g / 2 - b); r[0] = np.sqrt(h[0]) * z[0]
    for t in range(1, n):
        h[t] = om + (a + g * (r[t - 1] < 0)) * r[t - 1] ** 2 + b * h[t - 1]
        mu = case["mu0"] + case["mu1"] * h[t]           # daily conditional mean, percent (mu1 = risk-return slope)
        r[t] = mu + np.sqrt(h[t]) * z[t]
    rd = r[1:]
    ex = rd.reshape(n_months, D).sum(1) / 100
    rv = (rd.reshape(n_months, D) ** 2).sum(1) / 1e4
    origin = np.arange(n_months) * D - 1
    origin[0] = 0
    return rd, ex, rv, origin


def fit_all(rd, ex, rv, origin, gjr):
    h0 = rd.var()
    q = qmle(rd, gjr, h0)
    free = [1, 2, 3] if gjr else [1, 3]
    org = origin[1:]; exm, rvm = ex[1:], rv[1:]

    def w_of(th):
        v = monthly_vhat(th, rd, h0, org)
        if not (np.isfinite(v).all() and (v > 0).all()):
            return None
        return exposure(v, solve_c(v, W_MAX), W_MAX)

    def util(th):
        w = w_of(th)
        if w is None: return 1e10
        z = w * exm
        return -(z.mean() * 12 - GAMMA / 2 * z.var(ddof=1) * 12)

    def risk(th):
        w = w_of(th)
        return 1e10 if w is None else np.mean(rvm * w ** 2)

    out = {"qmle": q}
    for name, f in (("util", util), ("risk", risk)):
        obj = lambda p, f=f: (lambda th: 1e10 if not valid(th) else f(th))(np.r_[q[0], p[0], p[1] if gjr else 0.0, p[-1]])
        res = minimize(obj, q[free], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 800})
        res = minimize(obj, res.x, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10, "maxiter": 800})
        out[name] = np.r_[q[0], res.x[0], res.x[1] if gjr else 0.0, res.x[-1]]
    return out, h0


def evaluate(th, h0_train, rd_all, ex_all, rv_all, origin_all, cal_slice, test_slice):
    v = monthly_vhat(th, rd_all, h0_train, origin_all[1:])
    c = solve_c(v[cal_slice], W_MAX)
    w = exposure(v, c, W_MAX)[test_slice]
    x = w / w.mean()
    R, Q = ex_all[1:][test_slice], rv_all[1:][test_slice]
    z = x * R
    return {"logVR": np.log(np.mean(Q * x ** 2) / Q.mean()), "SR": z.mean() / z.std() * np.sqrt(12),
            "SR_un": R.mean() / R.std() * np.sqrt(12), "MT": z.mean() / R.mean()}


CASES = {
    "A correct GARCH, constant mean": {"theta": (0.012, 0.08, 0.0, 0.905), "mu0": 0.03, "mu1": 0.0, "fit_gjr": False},
    "B GJR truth, GARCH fitted": {"theta": (0.012, 0.02, 0.12, 0.905), "mu0": 0.03, "mu1": 0.0, "fit_gjr": False},
    "C GARCH + risk-return slope": {"theta": (0.012, 0.08, 0.0, 0.905), "mu0": 0.0, "mu1": 0.035, "fit_gjr": False},
}

if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    which = sys.argv[2:] or list(CASES)
    rows = []
    for cname in which:
        case = CASES[cname]
        for rep in range(reps):
            rng = np.random.default_rng(1000 + rep)
            rd, ex, rv, origin = sim(case, TRAIN_M + TEST_M, rng)
            n_tr = TRAIN_M * D
            fits, h0 = fit_all(rd[:n_tr], ex[:TRAIN_M], rv[:TRAIN_M], origin[:TRAIN_M], case["fit_gjr"])
            for k, th in fits.items():
                e = evaluate(th, h0, rd, ex, rv, origin, slice(0, TRAIN_M - 1), slice(TRAIN_M - 1, None))
                rows.append({"case": cname, "rep": rep, "loss": k, "alpha": th[1], "beta": th[3],
                             "rho": persistence(th), "a_over_om": th[1] / th[0], **e})
        print(cname, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("dfl_diagnosis/mc_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.groupby(["case", "loss"])[["rho", "a_over_om", "logVR", "SR", "SR_un", "MT"]].agg(["mean", "std"]).round(3).to_string())
    for cname in df.case.unique():
        d = df[df.case == cname].pivot(index="rep", columns="loss", values=["logVR", "SR"])
        print("\n", cname)
        for A, Bn in (("risk", "qmle"), ("util", "qmle"), ("risk", "util")):
            dv = d["logVR"][A] - d["logVR"][Bn]; ds = d["SR"][A] - d["SR"][Bn]
            print(f"  {A}-{Bn}: dlogVR mean {dv.mean():+.4f} (se {dv.std()/np.sqrt(len(dv)):.4f}, {int((dv<0).sum())}/{len(dv)} lower) | "
                  f"dSR mean {ds.mean():+.4f} (se {ds.std()/np.sqrt(len(ds)):.4f}, {int((ds>0).sum())}/{len(ds)} higher)")
