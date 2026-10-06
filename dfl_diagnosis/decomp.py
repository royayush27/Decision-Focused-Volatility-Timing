import os, sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1])); sys.path.insert(0, str(_P(__file__).resolve().parent))
os.chdir(_P(__file__).resolve().parents[1])   # run from anywhere; data paths are relative to the project root
import numpy as np, pandas as pd
from calibrate_c import solve_c, exposure
base=pd.read_csv("results_1970/market_forecasts.csv",index_col="month")
rv, R = base["rv"], base["ret_m"]
F={"Lagged RV":base["rv_baseline_forecast"]}
for n in ["garch_stat","gjr_stat","garch_dec","gjr_dec"]:
    F[n]=pd.read_csv(f"results_1970/forecasts/{n}.csv",index_col="month")["v_hat"]
W={k:pd.Series(exposure(v,solve_c(v.loc[:"1999-12"])),index=v.index) for k,v in F.items()}
def decomp(w, a, b):
    w,r,q=w.loc[a:b],R.loc[a:b],rv.loc[a:b]
    x=w/w.mean()                                   # normalise to mean exposure 1
    MT=(x*r).mean()/r.mean()                       # mean-timing factor
    VT=r.std()/(x*r).std()                         # variance-timing factor (sd version)
    VR=(q*x**2).mean()/q.mean()                     # variance ratio using realised variance (robust)
    SRu=r.mean()/r.std()*np.sqrt(12); SRm=(x*r).mean()/(x*r).std()*np.sqrt(12)
    return dict(SR_u=SRu,SR_m=SRm,ratio=SRm/SRu,MT=MT,VT=VT,check=MT*VT,VR_rv=VR,sqrt1_over_VR=1/np.sqrt(VR),corr_x_r=np.corrcoef(x,r)[0,1])
for lab,(a,b) in {"cal 1970-1999":("1970-01","1999-12"),"dev 2000-2024":("2000-01","2024-12"),"full 1970-2024":("1970-01","2024-12")}.items():
    print("==",lab,"  (SR ratio = MT x VT exactly; VR_rv = managed/unmanaged variance using realised variance)")
    print(pd.DataFrame({k:decomp(w,a,b) for k,w in W.items()}).T.round(3).to_string())
