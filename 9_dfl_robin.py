"""9. Decision-focused training (Robin's Experiment 1) under our frozen protocol.

Robin trained each variance model in two ways and saved the real-time monthly forecasts (robin_experiment/, data/robin/):
  objective A = accuracy: daily Gaussian QMLE
  objective B = decision: monthly utility of the managed portfolio after costs (starts from A, early stopping on the last
                20% of training months, mean held at its A value)
Models: GARCH(1,1), GJR-GARCH, and the Zhao et al. (2024) GARCH-LSTM (5 seeds per refit; the ensemble averages the seeds'
exposures). Plus 'MM on RV' (last month's realised variance). Factors: Mkt, SMB, HML, Mom, RMW, CMA (Ken French, US).
Every model was refit each January on data before that year (expanding window); first trade after 10 years of data
(1937 for Mkt, SMB, HML, Mom; 1974 for RMW, CMA).

Part 0 (check): Robin's own rule (his c, no cap, 10 bp, his full sample) must reproduce the Sharpe ratios in his report.
Part 1: his forecasts, OUR rule: c re-set every January (vol match on past months), cap 1.5, 10 bp; our periods; tests:
  each model vs buy-and-hold; his questions Q1 network (Zhao A - GJR A), Q2 decision training (B - A for GARCH, GJR, Zhao),
  Q3 full package (Zhao B - GJR A, Zhao B - MM on RV); Cederburg combination; Romano-Wolf over the 36 Q-tests on 1998-2025.
Part 2: forecast accuracy (QLIKE vs realised variance) of A vs B: does decision training make the forecast worse?
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from core import data, report
from core.rule import mm_weights, run
from core.stats import sharpe, boot_idx, sr_rows, dsr_test, fmt, cederburg, holm, bh, romano_wolf

S = report.start("9_dfl_robin", "9. Decision-focused training (Robin's Experiment 1) under our protocol")
FAC = ["Mkt", "SMB", "HML", "Mom", "RMW", "CMA"]
MODELS = ["RV", "GARCH_A", "GARCH_B", "GJR_A", "GJR_B", "ZHAO_A", "ZHAO_B"]
KEY = {"RV": ["RV"], "GARCH_A": ["garch_A"], "GARCH_B": ["garch_B"], "GJR_A": ["gjr_A"], "GJR_B": ["gjr_B"],
       "ZHAO_A": [f"zhao_A_s{i}" for i in range(5)], "ZHAO_B": [f"zhao_B_s{i}" for i in range(5)]}
TESTS = [("Q1 network", "ZHAO_A", "GJR_A"), ("Q2 decision", "ZHAO_B", "ZHAO_A"), ("Q2 decision", "GJR_B", "GJR_A"),
         ("Q2 decision", "GARCH_B", "GARCH_A"), ("Q3 full package", "ZHAO_B", "GJR_A"), ("Q3 full package", "ZHAO_B", "RV")]
REPORTED = {  # Robin's report (compare_output.txt), net Sharpe, full out-of-sample to Aug 2026, his rule
    "Mkt": [0.517, 0.405, 0.438, 0.454, 0.447, 0.443, 0.454, 0.393], "SMB": [0.165, 0.045, 0.040, 0.111, 0.076, 0.088, 0.092, 0.278],
    "HML": [0.388, 0.262, 0.371, 0.342, 0.376, 0.386, 0.366, 0.471], "Mom": [0.504, 0.647, 0.848, 0.824, 0.849, 0.826, 0.901, 0.937],
    "RMW": [0.414, 0.536, 0.505, 0.502, 0.516, 0.535, 0.530, 0.653], "CMA": [0.441, 0.297, 0.387, 0.367, 0.369, 0.373, 0.371, 0.435]}
rf = data.rf()
G = {f: pd.read_csv(config.DATA / "robin" / f"{f}_forecasts.csv", index_col="month") for f in FAC}

# ---------------------------------------------------------------- Part 0: reproduce Robin's numbers with his own rule
print("Part 0. Check: Robin's own rule (his c, no cap, 10 bp, full sample to Aug 2026) vs his reported net Sharpe")
rows = {}
for f, g in G.items():
    out = [sharpe(g["f"])]
    for m in MODELS:
        w = sum(g[f"c_{k}"] / g[f"V_{k}"] for k in KEY[m]) / len(KEY[m])
        out.append(sharpe(w * g["f"] - 0.001 * w.diff().abs().fillna(0)))
    rows[f] = out
cols = ["BH"] + MODELS
P0 = pd.DataFrame(rows, index=cols).T[["BH", "RV", "GARCH_A", "GARCH_B", "GJR_A", "GJR_B", "ZHAO_A", "ZHAO_B"]]
D0 = P0 - pd.DataFrame(REPORTED, index=P0.columns).T
report.show(P0, 3)
print(f"largest absolute difference from his report: {np.abs(D0.to_numpy()).max():.4f}\n")

# ---------------------------------------------------------------- Part 1: our rule
nets = {}
for f, g in G.items():
    g = g.loc[:"2025-12"]; e = g["f"]; start = g.index[0]
    nets[(f, "BH")] = e
    for m in MODELS:
        ws = [mm_weights(g[f"V_{k}"], e, S, (start, "1997-12"))[0] for k in KEY[m]]
        w = sum(ws) / len(ws)
        nets[(f, m)] = run(w, e, rf, S["cost"])[0]
first = {f: max(z.index[0] for (ff, m), z in nets.items() if ff == f and m != "BH") for f in FAC}
PER = {"full sample": (None, "2025-12"), **config.PERIODS}
for p, (a, b) in PER.items():
    T, Q = {}, {}
    for f in FAC:
        aa = a or first[f]
        u = nets[(f, "BH")].loc[aa:b]; I = boot_idx(len(u))
        T[f] = {"BH": sharpe(u), **{m: sharpe(nets[(f, m)].loc[aa:b]) for m in MODELS}}
        for q, x, y in TESTS:
            d, lo, hi, pv, _ = dsr_test(nets[(f, x)].loc[aa:b], nets[(f, y)].loc[aa:b], I)
            Q.setdefault(f"{q}: {x} - {y}", {})[f] = fmt(d, lo, hi)
        for m in ("GJR_A", "ZHAO_A", "ZHAO_B"):
            d, lo, hi, pv, _ = dsr_test(nets[(f, m)].loc[aa:b], u, I)
            Q.setdefault(f"vs BH: {m}", {})[f] = fmt(d, lo, hi)
    label = f"{p} (from {first['Mkt']} / {first['RMW']} for RMW, CMA)" if a is None else p
    print(f"======== {label}: net Sharpe under our rule")
    report.show(pd.DataFrame(T).T)
    print("Sharpe differences [95% CI]"); report.show(pd.DataFrame(Q).T)

print("Cederburg real-time combination gain vs buy-and-hold [95% CI], 1998-2025")
a, b = config.PERIODS[config.MAIN]; C = {}
for f in FAC:
    u = nets[(f, "BH")]
    for m in ("RV", "GJR_A", "ZHAO_A", "ZHAO_B"):
        c = cederburg(u, nets[(f, m)])[0].loc[a:b]; d, lo, hi, pv, _ = dsr_test(c, u.loc[a:b])
        C.setdefault(m, {})[f] = fmt(d, lo, hi)
report.show(pd.DataFrame(C).T)

print(f"Family of the 36 Q-tests on {config.MAIN}: one joint bootstrap (same months for every test), Holm / BH / Romano-Wolf")
n = len(nets[("Mkt", "BH")].loc[a:b]); I = boot_idx(n); rows, draws = [], []
for f in FAC:
    for q, x, y in TESTS:
        A, B = nets[(f, x)].loc[a:b].to_numpy(), nets[(f, y)].loc[a:b].to_numpy()
        d = sharpe(A) - sharpe(B); db = sr_rows(A[I]) - sr_rows(B[I]); se = db.std(ddof=1)
        rows.append({"question": q, "factor": f, "test": f"{x} - {y}", "dSR": d, "t": d / se, "p_raw": 2 * (1 - norm.cdf(abs(d / se)))})
        draws.append((db - d) / se)
F = pd.DataFrame(rows); F["p_holm"], F["p_bh"] = holm(F.p_raw), bh(F.p_raw); F["p_romano_wolf"] = romano_wolf(F.t, np.vstack(draws))
F.to_csv(config.out_dir() / "9_dfl_robin_family.csv", index=False)
report.show(F.sort_values("p_raw").head(12), 3)

# ---------------------------------------------------------------- Part 2: forecast accuracy
print(f"Part 2. Forecast accuracy on {config.MAIN}: mean QLIKE (lower = better) vs realised variance; ratio B / A")
A2 = {}
for f, g in G.items():
    g = g.loc[a:b]
    ql = lambda k: float(np.mean(g["rv"] / g[f"V_{k}"] - np.log(g["rv"] / g[f"V_{k}"]) - 1))
    zA = np.mean([ql(k) for k in KEY["ZHAO_A"]]); zB = np.mean([ql(k) for k in KEY["ZHAO_B"]])
    A2[f] = {"GARCH A": ql("garch_A"), "GARCH B": ql("garch_B"), "GJR A": ql("gjr_A"), "GJR B": ql("gjr_B"),
             "Zhao A (seed mean)": zA, "Zhao B (seed mean)": zB, "Zhao B / A": zB / zA}
report.show(pd.DataFrame(A2).T, 3)
report.save_nets("9_dfl_robin", {("Robin DFL", f, "Unmanaged" if m == "BH" else m): z for (f, m), z in nets.items()})
