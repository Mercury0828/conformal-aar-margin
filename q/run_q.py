"""Confirmatory analyses Q1–Q3 and Q5. Usage: python q/run_q.py val|test

Writes results/Q1_<period>.csv, Q2_<period>.csv, Q3_<period>.csv, Q5_<period>.csv and Q_manifest_<period>.json.
The "test" run is the confirmatory result (first run of the frozen method).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def subsets(ev):
    yield "all", np.ones(len(ev), bool)
    yield "heatwave", ev["HW"].values
    yield "before_B6", ~ev["after_B6"].values
    yield "after_B6", ev["after_B6"].values


def q1(ev):
    rows = []
    for case in C.CASES:
        key = f"{case[0]}_{int(case[1])}"
        rel = (ev[f"R_hat_{key}"] - ev[f"R_obs_{key}"]) / ev[f"R_obs_{key}"]
        over = (ev[f"R_hat_{key}"] > ev[f"R_obs_{key}"]).astype(float)
        for sname, m in subsets(ev):
            for band in C.BAND_ORDER:
                mb = m & (ev["band"].values == band)
                if not mb.any():
                    continue
                g = ev[mb]
                one = pd.Series(1.0, index=g.index)
                r = C.ratio_bootstrap(over[mb], one, g["day"])
                t5 = C.ratio_bootstrap((rel[mb] > 0.05).astype(float), one, g["day"])  # day blocks
                rows.append({"case": key, "subset": sname, "band": band, "n": int(mb.sum()), "rate": r[0],
                             "rate_lo": r[1], "rate_hi": r[2], "p95_rel": float(np.percentile(rel[mb], 95)),
                             "share_gt5": t5[0], "share_gt5_lo": t5[1], "share_gt5_hi": t5[2],
                             "mean_e": float(g["e"].mean())})
    df = pd.DataFrame(rows)
    p = df[(df.case == "Drake_75") & (df.subset == "all")]
    df.attrs["bar_every_band"] = bool(((p.rate > 0.10) & (p.p95_rel > 0.01)).all())
    return df


def q2_q3(ev, hist):
    rows, timing = [], {}
    for alpha in C.ALPHAS:
        t0 = time.time()
        q_split = C.bounds(ev, hist, alpha)
        timing[f"split_{alpha}"] = time.time() - t0
        t0 = time.time()
        q_aci = C.bounds(ev, hist, alpha, aci_gamma=0.001)
        timing[f"aci_{alpha}"] = time.time() - t0
        margins = {"AARGUARD": q_split, "AARGUARD-ACI": q_aci}
        bl = C.frozen_baselines(ev, hist, alpha)
        for b in bl.columns:
            margins[b] = bl[b]
        r_los = {name: C.rating_at(ev, (ev["T_hat"] + q).values, *C.PRIMARY) for name, q in margins.items()}
        for name, r_lo in r_los.items():
            for sname, m in subsets(ev):
                for band in C.BAND_ORDER:
                    mb = m & (ev["band"].values == band)
                    if not mb.any():
                        continue
                    met = C.metrics(ev[mb], r_lo[mb])
                    row = {"method": name, "alpha": alpha, "subset": sname, "band": band, **met,
                           "meets_Q2_tol": met["cov"] >= C.tol(alpha)}
                    if sname == "all":
                        w7 = C.metrics(ev[mb], r_lo[mb], block="week")
                        row.update({"cov_lo_7d": w7["cov_lo"], "cov_hi_7d": w7["cov_hi"], "cov_p5_7d": w7["cov_p5"]})
                    if name != "AARGUARD":
                        pr = C.paired(ev[mb], r_los["AARGUARD"][mb], r_lo[mb])
                        row.update({"AARGUARD_minus_this_d_cov": pr["d_cov"], "d_cov_lo": pr["d_cov_lo"],
                                    "d_cov_hi": pr["d_cov_hi"], "AARGUARD_minus_this_d_CC": pr["d_CC"],
                                    "d_CC_lo": pr["d_CC_lo"], "d_CC_hi": pr["d_CC_hi"]})
                    rows.append(row)
        r_stat = ev["R_stat_Drake_75"].values
        for sname, m in subsets(ev):
            for band in C.BAND_ORDER:
                mb = m & (ev["band"].values == band)
                if mb.any():
                    rows.append({"method": "B4", "alpha": alpha, "subset": sname, "band": band,
                                 **C.metrics(ev[mb], r_stat[mb]), "meets_Q2_tol": None})
        print("alpha done", alpha, flush=True)
    return pd.DataFrame(rows), timing


def per_station(ev, hist):
    q = C.bounds(ev, hist, 0.05)
    r_lo = C.rating_at(ev, (ev["T_hat"] + q).values, *C.PRIMARY)
    cov = (r_lo <= ev["R_obs_Drake_75"].values + 1e-9)
    return (ev.assign(cov=cov).groupby(["station", "band"])["cov"].agg(["size", "mean"]).reset_index())


def q5(ev, hist):
    rows = []
    for w in (90, 365, 730):
        for cells in ("season", "cell"):
            q = C.bounds(ev, hist, 0.05, w=w, cells=cells)
            for case in C.CASES:
                r_lo = C.rating_at(ev, (ev["T_hat"] + q).values, *case)
                for band in C.BAND_ORDER:
                    mb = ev["band"].values == band
                    rows.append({"W": w, "cells": cells, "case": f"{case[0]}_{int(case[1])}", "band": band,
                                 **C.metrics(ev[mb], r_lo[mb], case=case)})
    for gm in (0.001, 0.005):
        q = C.bounds(ev, hist, 0.05, aci_gamma=gm)
        r_lo = C.rating_at(ev, (ev["T_hat"] + q).values, *C.PRIMARY)
        for band in C.BAND_ORDER:
            mb = ev["band"].values == band
            rows.append({"W": 90, "cells": f"cell+ACI{gm}", "case": "Drake_75", "band": band, **C.metrics(ev[mb], r_lo[mb])})
    return pd.DataFrame(rows)


def main(period: str) -> None:
    t_all = time.time()
    ev, hist = C.load(period)
    print(period, "rows", len(ev), "heat-wave rows", int(ev["HW"].sum()), "stations", ev["station"].nunique(), flush=True)
    d1 = q1(ev); C.save(d1, "Q1", period)
    d23, timing = q2_q3(ev, hist); C.save(d23, "Q2Q3", period)
    ps = per_station(ev, hist); C.save(ps, "Q2_station", period)
    d5 = q5(ev, hist); C.save(d5, "Q5", period)
    man = C.manifest(period)
    station_years = ev["station"].nunique() * (ev["valid"].max() - ev["valid"].min()).days / 365.25
    man.update({**C.COUNTS,
                "rows": len(ev), "heatwave_rows": int(ev["HW"].sum()), "seconds_total": time.time() - t_all,
                "seconds_per_method": timing, "station_years": station_years,
                "split_seconds_per_station_year_alpha05": timing["split_0.05"] / station_years,
                "Q1_bar_every_band": d1.attrs.get("bar_every_band")})
    (C.RES / f"Q_manifest_{period}.json").write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps(man, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
