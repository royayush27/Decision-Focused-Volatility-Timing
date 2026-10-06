"""Load Kenneth French factor files (monthly + daily) exactly as Moreira & Muir (2017) use them.

Factors: Mkt-RF, SMB, HML, Mom, RMW, CMA. Returns are converted from percent to decimals.
"""
import io
from pathlib import Path
import pandas as pd

DATA = Path(__file__).parent / "data"
FACTORS = ["Mkt", "SMB", "HML", "Mom", "RMW", "CMA"]


def _read_block(fname, daily):
    """Read the first data block of a French CSV (skips header text and the annual block)."""
    lines = (DATA / fname).read_text(encoding="latin-1").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))  # column header line
    rows = [lines[start]]
    ndig = 8 if daily else 6
    for l in lines[start + 1:]:
        key = l.split(",")[0].strip()
        if len(key) == ndig and key.isdigit():
            rows.append(l)
        else:
            break
    df = pd.read_csv(io.StringIO("\n".join(rows)), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d" if daily else "%Y%m")
    if not daily:
        df.index = df.index.to_period("M")
    df = df.astype(float) / 100.0
    df[df <= -0.9999] = float("nan")  # French uses -99.99 as missing
    return df


def load(daily):
    s = "_daily" if daily else ""
    ff3 = _read_block(f"F-F_Research_Data_Factors{s}.csv", daily)
    ff5 = _read_block(f"F-F_Research_Data_5_Factors_2x3{s}.csv", daily)
    mom = _read_block(f"F-F_Momentum_Factor{s}.csv", daily)
    df = pd.DataFrame({
        "Mkt": ff3["Mkt-RF"], "SMB": ff3["SMB"], "HML": ff3["HML"],
        "Mom": mom.iloc[:, 0], "RMW": ff5["RMW"], "CMA": ff5["CMA"], "RF": ff3["RF"],
    })
    return df


if __name__ == "__main__":
    for d in (False, True):
        df = load(d)
        print("daily" if d else "monthly", df.shape)
        print(df.apply(lambda s: (s.first_valid_index(), s.last_valid_index())).T)
