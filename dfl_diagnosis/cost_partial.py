import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import numpy as np, pandas as pd
from backtest import simulate
from calibrate_c import exposure, solve_c
from risk_dfl import FACTORS
# Is the cost-aware decision loss better than simply smoothing the QMLE weights? w_t = w_{t-1} + lam (w*_t - w_{t-1})
SERIES=["Mkt-RF","SMB","HML","RMW","CMA","Mom","NoDur","Durbl","Manuf","Enrgy","HiTec","Telcm","Shops","Hlth","Utils","Other"]
m=pd.read_csv("data/ff_monthly_baselinedata.csv"); rf_all=pd.read_csv("data/ff_monthly_rf.csv",index_col="month")["RF"]
rng=np.random.default_rng(2); B,BL=2000,6
for kbp in (25,50):
    kappa=kbp/1e4; nets={}; rows=[]
    for s in SERIES:
        ms=m[m.series==s].set_index("month").sort_index().loc["1970-01":"2025-12"]; rf=rf_all.reindex(ms.index)
        ex=ms["ret_m"] if s in FACTORS else ms["ret_m"]-rf; rv=ms["rv"]
        d=pd.read_csv(f"dfl_diagnosis/panel/{s}_cost{kbp}.csv",index_col="month")
        wq=exposure(d["qmle"],solve_c(d["qmle"].loc[:"1999-12"])); wc=exposure(d["cost_risk"],solve_c(d["cost_risk"].loc[:"1999-12"]))
        rules={"qmle":wq,"cost_risk":wc}
        for lam in (0.5,0.25,0.1):
            w=np.empty_like(wq); w[0]=wq[0]
            for t in range(1,len(w)): w[t]=w[t-1]+lam*(wq[t]-w[t-1])
            rules[f"qmle+partial {lam}"]=w
        for k,w in rules.items():
            sim=simulate(pd.Series(w,index=ex.index),ex,rf,kappa); sim.index=ex.index
            z=sim.net_ex.loc["2000-01":"2024-12"]; W=pd.Series(w,index=ex.index).loc["2000-01":"2024-12"]; x=W/W.mean(); Q=rv.loc["2000-01":"2024-12"]
            nets[(k,s)]=z.to_numpy()
            rows.append({"rule":k,"series":s,"SR":z.mean()/z.std()*np.sqrt(12),"turnover":sim.turnover.loc["2000-01":"2024-12"].mean(),"VR":(Q*x**2).mean()/Q.mean()})
    T=pd.DataFrame(rows); print(f"\n== {kbp} bps"); print(T.groupby("rule")[["SR","turnover","VR"]].mean().round(4).to_string())
    n=len(next(iter(nets.values()))); st=rng.integers(0,n,(B,int(np.ceil(n/BL)))); I=((st[:,:,None]+np.arange(BL))%n).reshape(B,-1)[:,:n]
    for Bn in ("qmle+partial 0.5","qmle+partial 0.25","qmle+partial 0.1"):
        dB=np.zeros(B); d0=[]
        for s in SERIES:
            a,b=nets[("cost_risk",s)],nets[(Bn,s)]
            dB+=(a[I].mean(1)/a[I].std(1,ddof=1)-b[I].mean(1)/b[I].std(1,ddof=1))*np.sqrt(12); d0.append((a.mean()/a.std(ddof=1)-b.mean()/b.std(ddof=1))*np.sqrt(12))
        dB/=16; m0=np.mean(d0)
        print(f"  cost_risk - {Bn}: dSR {m0:+.4f} CI [{np.percentile(dB-dB.mean()+m0,2.5):+.4f},{np.percentile(dB-dB.mean()+m0,97.5):+.4f}] p={np.mean(np.abs(dB-dB.mean())>=abs(m0)):.3f} better {int(np.sum(np.array(d0)>0))}/16")
