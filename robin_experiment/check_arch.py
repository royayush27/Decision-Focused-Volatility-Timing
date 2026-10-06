"""Validate our GARCH/GJR QMLE against the arch package (Mkt, 1926-2015)."""
import time, numpy as np, torch
from arch import arch_model
from data import load
from models import VolModel
from train import fit_A_garch
d = load(True)["Mkt"].loc[:"2015-12-31"].dropna()
r = torch.tensor(d.values)
for kind, o in [("garch", 0), ("gjr", 1)]:
    t = time.time()
    m = VolModel(kind, d.var()); ll = fit_A_garch(m, r)
    print(kind, "ours", {k: round(v, 6) for k, v in m.readable().items()}, "loss", ll, f"{time.time()-t:.1f}s")
    am = arch_model(d.values * 100, mean="Constant", vol="GARCH", p=1, o=o, q=1).fit(disp="off")
    print("   arch", am.params.round(6).to_dict(), "omega(dec)", am.params["omega"] / 1e4,
          "loglik", am.loglikelihood)
    # our loss -> comparable loglik: -0.5*sum(log2pi + log h + e2/h), in percent units
    n = len(d) - 1
    print("   ours loglik(pct units)", -0.5 * n * (np.log(2 * np.pi) + ll + 2 * np.log(100)) )
