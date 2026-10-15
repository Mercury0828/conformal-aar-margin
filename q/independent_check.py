"""Independent re-computation: split-conformal coverage for one station x band, written from scratch with
plain loops (no aarguard.bound / rating code; IEEE 738 not needed because coverage is evaluated on temperature, which is
equivalent under the verified monotone transfer with ties covered). Usage: python q/independent_check.py val|test KAUS 25-48"""
import math, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]
period, station, band = sys.argv[1], sys.argv[2], sys.argv[3]
tag = "OPEN" if period == "test" else "closed"
d = pd.read_parquet(REPO / "data" / "aligned" / f"aligned_GFS_12Z_{station}_{tag}.parquet")
lo_p, hi_p = (("2021-06-01", "2022-06-01") if period == "val" else ("2022-06-01", "2025-06-01"))
d = d[d["band"] == band]
hist = d.dropna(subset=["e", "t_obs"])
ev = d[(d["valid"] >= pd.Timestamp(lo_p, tz="UTC")) & (d["valid"] < pd.Timestamp(hi_p, tz="UTC"))].dropna(subset=["T"])
covered, total = 0, 0
for _, r in ev.iterrows():
    tau = r["issue"]
    def pick(days):
        m = ((hist["cell"] == r["cell"]) & (hist["valid"] > tau - pd.Timedelta(days=days)) & (hist["valid"] <= tau)
             & (hist["t_obs"] <= tau))
        return np.sort(hist.loc[m, "e"].values)
    e = pick(90)
    if math.ceil((len(e) + 1) * 0.95) > len(e):
        e = pick(730)
    k = math.ceil((len(e) + 1) * 0.95)
    q = e[k - 1] if k <= len(e) else math.inf
    covered += int(r["T"] <= r["T_hat"] + q + 1e-9); total += 1
print(f"independent coverage {station} {band} alpha 0.05 ({period}): {covered}/{total} = {covered/total:.6f}")
