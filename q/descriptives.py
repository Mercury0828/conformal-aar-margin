"""Registered descriptive reports that q/run_q.py did not write: Q1 p95 CIs and the station x
band x season exposure table; Q2 per-season and per-lead coverage of the frozen split / ACI bounds. Same frozen method and
data; no new analysis. Usage: python q/descriptives.py test"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from aarguard.evalstats import bootstrap  # noqa: E402

period = sys.argv[1]
ev, hist = C.load(period)
key = "Drake_75"
rel = (ev[f"R_hat_{key}"] - ev[f"R_obs_{key}"]) / ev[f"R_obs_{key}"]
ev = ev.assign(rel=rel.values, over=(ev[f"R_hat_{key}"] > ev[f"R_obs_{key}"]).astype(float))
rows = []
for sname, m in (("all", np.ones(len(ev), bool)), ("heatwave", ev["HW"].values)):
    for band in C.BAND_ORDER:
        g = ev[m & (ev["band"].values == band)]
        p = bootstrap(g, lambda x: np.percentile(x["rel"], 95))
        rows.append({"subset": sname, "band": band, "n": len(g), "p95_rel": p[0], "p95_lo": p[1], "p95_hi": p[2]})
pd.DataFrame(rows).to_csv(C.RES / f"Q1_p95ci_{period}.csv", index=False)
tab = ev.groupby(["station", "band", "season"]).agg(n=("rel", "size"), rate=("over", "mean"),
                                                    p95_rel=("rel", lambda x: np.percentile(x, 95)),
                                                    share_gt5=("rel", lambda x: float((x > 0.05).mean())),
                                                    mean_e=("e", "mean")).reset_index()
tab.to_csv(C.RES / f"Q1_station_band_season_{period}.csv", index=False)
out = []
for alpha in C.ALPHAS:
    for name, q in (("AARGUARD", C.bounds(ev, hist, alpha)), ("AARGUARD-ACI", C.bounds(ev, hist, alpha, aci_gamma=0.001))):
        r_lo = C.rating_at(ev, (ev["T_hat"] + q).values, *C.PRIMARY)
        cov = (r_lo <= ev[f"R_obs_{key}"].values + 1e-9).astype(float)
        e2 = ev.assign(cov=cov)
        for (b, s), g in e2.groupby(["band", "season"]):
            out.append({"method": name, "alpha": alpha, "kind": "season", "band": b, "group": s, "n": len(g), "cov": g["cov"].mean()})
        for l, g in e2.groupby("lead"):
            out.append({"method": name, "alpha": alpha, "kind": "lead", "band": g["band"].iloc[0], "group": int(l), "n": len(g),
                        "cov": g["cov"].mean()})
pd.DataFrame(out).to_csv(C.RES / f"Q2_season_lead_{period}.csv", index=False)
d = pd.DataFrame(out)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(d[(d.alpha == 0.05) & (d.method == "AARGUARD")].groupby("kind")["cov"].agg(["min", "max"]).round(4))
