"""Q4 heat-wave case studies. Usage: python q/q4_case.py val|test

Statewide heat-wave days: >= 6 of the 12 stations are station heat-wave days (calendar-complete rule), matched by local
standard date. Runs ranked by length, then mean station-average daily maximum (hourly observations), then earlier start.
One station for both runs: most distinct observed heat-wave hours (hourly observations) across their union, ties -> station
priority. Trace: split AARGUARD (alpha 0.05, Drake/75), unique 12 UTC issue with 25 <= valid - issue <= 48 h, gaps kept.
Writes results/Q4_runs_<period>.csv and results/Q4_trace_<period>.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from aarguard import align, obs  # noqa: E402
from aarguard.heat import heatwave_days  # noqa: E402


def main(period: str) -> None:
    ev, hist = C.load(period)
    a, b = C.PERIODS[period]
    st_meta = align.stations()
    p95 = pd.read_csv(C.RES / "D1_heatwave.csv").set_index("icao")["p95_JJA_2018_2020"]
    hw_sets, dmax, hourly = {}, {}, {}
    for s in C.stations():
        o = obs.read_isd(s, range(2018, 2026))
        off = int(st_meta.loc[s, "lst_offset_h"])
        hw_sets[s], _ = heatwave_days(o, p95[s], off)
        lst = (o["time"] + pd.Timedelta(hours=off)).dt.tz_localize(None)
        o = o.assign(date=lst.dt.normalize().values)
        o = o[(o["time"] >= a) & (o["time"] < b)]
        dmax[s] = o.groupby("date")["T"].max()
        hourly[s] = o.assign(hw=o["date"].isin(hw_sets[s]) & (o["T"] > p95[s]))
    dates = pd.date_range(min(d.index.min() for d in dmax.values()), max(d.index.max() for d in dmax.values()), freq="D")
    count = pd.Series(0, index=dates)
    for s in C.stations():
        count += pd.Series(dates.isin(list(hw_sets[s])), index=dates).astype(int)
    statewide = count[count >= 6].index
    avg = pd.concat(dmax, axis=1).reindex(dates)
    smax, n_avail = avg.mean(axis=1), avg.notna().sum(axis=1)
    runs, cur = [], []
    for d in statewide:
        if cur and (d - cur[-1]).days == 1:
            cur.append(d)
        else:
            if cur:
                runs.append(cur)
            cur = [d]
    if cur:
        runs.append(cur)
    runs = sorted(runs, key=lambda r: (-len(r), -float(smax.reindex(r).mean()), r[0]))[:2]
    union = set(d for r in runs for d in r)
    prio = pd.read_csv(C.RES / "D1_selected_stations.csv").set_index("icao")["priority"]
    hw_hours = {s: int(hourly[s][hourly[s]["date"].isin(union) & hourly[s]["hw"]]["time"].dt.floor("h").nunique())
                for s in C.stations()}
    station = sorted(hw_hours, key=lambda s: (-hw_hours[s], prio[s]))[0]
    q = C.bounds(ev, hist, 0.05)
    ev = ev.assign(R_lo=C.rating_at(ev, (ev["T_hat"] + q).values, *C.PRIMARY))
    full = align.build("GFS", station)
    full = full[(full["lead"] >= 25) & (full["lead"] <= 48)]
    rows, traces = [], []
    for i, r in enumerate(runs, 1):
        grid = pd.date_range(pd.Timestamp(r[0]).tz_localize("UTC") - pd.Timedelta(hours=int(st_meta.loc[station, "lst_offset_h"])),
                             periods=len(r) * 8, freq="3h").ceil("3h")
        tr = pd.DataFrame({"valid": grid}).merge(full[["valid", "issue", "lead"]], on="valid", how="left")
        tr = tr.merge(ev.loc[ev["station"] == station, ["valid", "issue", "T", "T_hat", "R_obs_Drake_75", "R_hat_Drake_75",
                                                        "R_lo", "R_stat_Drake_75", "HW"]], on=["valid", "issue"], how="left")
        tr = tr.assign(run=i, station=station)
        traces.append(tr)
        ok = tr["R_obs_Drake_75"].notna()
        rows.append({"run": i, "start": r[0], "end": r[-1], "days": len(r),
                     "mean_station_avg_daily_max": float(smax.reindex(r).mean()),
                     "min_stations_with_max": int(n_avail.reindex(r).min()), "station": station,
                     "station_hw_hours_union": hw_hours[station], "evaluated_cases": int(ok.sum()),
                     "cases_Rhat_gt_Robs": int((tr.loc[ok, "R_hat_Drake_75"] > tr.loc[ok, "R_obs_Drake_75"]).sum()),
                     "cases_Rlo_gt_Robs": int((tr.loc[ok, "R_lo"] > tr.loc[ok, "R_obs_Drake_75"] + 1e-9).sum()),
                     "gaps": int((~ok).sum())})
    pd.DataFrame(rows).to_csv(C.RES / f"Q4_runs_{period}.csv", index=False)
    if traces:
        pd.concat(traces).to_csv(C.RES / f"Q4_trace_{period}.csv", index=False)
    print(pd.DataFrame(rows).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
