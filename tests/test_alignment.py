"""Deterministic leakage and alignment tests."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aarguard import align, bound, obs  # noqa: E402


def utc(s):
    return pd.Timestamp(s, tz="UTC")


def test_obs_window_rule():
    o = pd.DataFrame({"time": [utc("2021-07-01 11:53"), utc("2021-07-01 12:04"), utc("2021-07-01 14:10")],
                      "T": [30.0, 31.0, 33.0]})
    v = pd.Series([utc("2021-07-01 12:00"), utc("2021-07-01 15:00"), utc("2021-07-01 14:00")])
    got = obs.at_valid_hours(o, v)
    assert got.iloc[0] == 31.0          # +4 min is nearer than -7 min
    assert math.isnan(got.iloc[1])      # nothing within [t-15, t+5]
    assert math.isnan(got.iloc[2])      # +10 min is too late


def _frame():
    issue = pd.date_range("2021-07-01 12:00", periods=3, freq="D", tz="UTC")
    rows = [(i, i + pd.Timedelta(hours=l), l) for i in issue for l in sorted(align.GFS_LEADS)]
    d = pd.DataFrame(rows, columns=["issue", "valid", "lead"])
    d["T_hat"] = 30.0
    d["T"] = 31.0
    return d


def test_grid_passes_and_detects_one_hour_valid_shift():
    d = _frame()
    align.check_grid(d)
    bad = d.copy()
    bad.loc[5, "valid"] += pd.Timedelta(hours=1)
    with pytest.raises(align.AlignmentError):
        align.check_grid(bad)


def test_detects_issue_time_shift():
    bad = _frame()
    bad["issue"] += pd.Timedelta(hours=1)
    with pytest.raises(align.AlignmentError):
        align.check_grid(bad)


def test_detects_fahrenheit_celsius_swap():
    d = _frame()
    align.check_units(d)
    bad = d.copy()
    bad["T_hat"] = bad["T_hat"] * 9 / 5 + 32  # deg F left unconverted
    with pytest.raises(align.AlignmentError):
        align.check_units(bad)


def test_calibration_excludes_errors_after_tau_and_guard_raises():
    tau = utc("2021-07-10 12:00")
    hist = pd.DataFrame({"station": "X", "band": "7-24", "cell": "JJA-2",
                         "valid": pd.date_range("2021-06-01", periods=400, freq="3h", tz="UTC"), "e": 1.0})
    hist["t_obs"] = hist["valid"] - pd.Timedelta(minutes=7)
    hist.loc[hist["t_obs"] > tau, "e"] = 100.0     # values reported after tau must never enter
    assert (hist["e"] == 100.0).sum() > 50         # the poison is really present
    tgt = pd.DataFrame({"station": ["X"], "band": ["7-24"], "cell": ["JJA-2"], "issue": [tau],
                        "valid": [tau + pd.Timedelta(hours=12)]})
    q = bound.split_bounds(hist, tgt, alpha=0.05, w_days=365)
    assert q.iloc[0] == 1.0
    with pytest.raises(bound.LeakageError):
        bound.assert_no_leakage(np.array([np.datetime64("2021-07-11T00:00")]), np.datetime64("2021-07-10T12:00"))


def test_conformal_quantile_and_small_cell():
    e = np.arange(1, 20, dtype=float)   # n = 19, alpha = 0.05 -> k = ceil(20 * 0.95) = 19
    assert bound.conformal_q(e, 0.05) == 19.0
    assert math.isinf(bound.conformal_q(e[:10], 0.05))  # k = 11 > n = 10
    assert math.isinf(bound.conformal_q(np.array([]), 0.05))


def test_dst_day_local_time_uses_fixed_offset():
    v = pd.Series(pd.date_range("2021-03-14 06:00", periods=4, freq="1h", tz="UTC"))
    lst = (v + pd.Timedelta(hours=-6)).dt.tz_localize(None)
    assert list(lst.dt.hour) == [0, 1, 2, 3]   # no DST gap at 02:00 in local standard time


def test_report_after_tau_with_valid_at_tau_is_excluded():
    """R1 audit counterexample: valid 12:00 = tau but the selected report is at 12:04 -> not available at tau."""
    tau = utc("2021-07-10 12:00")
    valid = list(pd.date_range("2021-06-20 12:00", periods=19, freq="D", tz="UTC")) + [tau]
    hist = pd.DataFrame({"station": "X", "band": "7-24", "cell": "JJA-2", "valid": valid,
                         "t_obs": [v - pd.Timedelta(minutes=7) for v in valid[:-1]] + [tau + pd.Timedelta(minutes=4)],
                         "e": [1.0] * 19 + [10.0]})
    tgt = pd.DataFrame({"station": ["X"], "band": ["7-24"], "cell": ["JJA-2"], "issue": [tau],
                        "valid": [tau + pd.Timedelta(hours=12)]})
    q = bound.split_bounds(hist, tgt, alpha=0.05, w_days=365, w_max_days=365)
    assert q.iloc[0] == 1.0     # n = 19 available errors -> k = 19 -> 1.0; with the leaked 10 degC error it would be 10.0


def test_test_period_guard_in_raw_loaders(monkeypatch):
    from aarguard import periods
    d = pd.DataFrame({"time": [utc("2022-05-31 23:53"), utc("2022-06-01 00:53")], "T": [30.0, 31.0]})
    out = periods.drop_test(d, ("time",))
    assert len(out) == 1 and out["T"].iloc[0] == 30.0


def test_window_lower_endpoint_uses_valid_time():
    """Regression: valid hour tau - W is outside the frozen window even if its report came 4 min later."""
    tau = utc("2021-07-10 12:00")
    edge = tau - pd.Timedelta(days=30)
    valid = [edge] + list(pd.date_range(edge + pd.Timedelta(hours=3), periods=19, freq="D", tz="UTC"))
    hist = pd.DataFrame({"station": "X", "band": "7-24", "cell": "JJA-2", "valid": valid,
                         "t_obs": [v + pd.Timedelta(minutes=4) for v in valid], "e": [50.0] + [1.0] * 19})
    tgt = pd.DataFrame({"station": ["X"], "band": ["7-24"], "cell": ["JJA-2"], "issue": [tau],
                        "valid": [tau + pd.Timedelta(hours=12)]})
    q, n = bound.split_bounds(hist, tgt, alpha=0.05, w_days=30, w_max_days=30, return_n=True)
    assert n.iloc[0] == 19 and q.iloc[0] == 1.0


def test_missing_date_breaks_heatwave_run():
    from aarguard.heat import heatwave_days
    t = [utc(f"2021-07-0{d} 21:00") for d in (1, 3, 4)]       # July 2 has no observation
    hw, n_missing = heatwave_days(pd.DataFrame({"time": t, "T": [40.0, 40.0, 40.0]}), 38.0, -6)
    assert hw == set() and n_missing == 1
    t = [utc(f"2021-07-0{d} 21:00") for d in (1, 2, 3)]
    hw, _ = heatwave_days(pd.DataFrame({"time": t, "T": [40.0, 40.0, 40.0]}), 38.0, -6)
    assert len(hw) == 3
