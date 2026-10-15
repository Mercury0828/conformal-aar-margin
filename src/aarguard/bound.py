"""M-bound: AARGUARD one-sided split-conformal temperature bound and its ACI variant.

For a target forecast (station s, lead band b, cell j, issue tau) the calibration set holds the errors e = T - T_hat of
forecasts at the same s, b, j with tau - W < valid hour <= tau (frozen window, valid time) whose observation report time
t_obs <= tau (availability; nothing reported after tau enters). With n = |K| and k = ceil((n+1)(1-alpha)):
q_hat = k-th smallest error if k <= n; otherwise the window is widened to W_max; if still k > n, q_hat = +inf (R_lo = 0).
T_plus = T_hat + q_hat.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

W_MAX_DAYS = 730


class LeakageError(AssertionError):
    pass


def conformal_q(sorted_errors: np.ndarray, alpha: float) -> float:
    n = len(sorted_errors)
    k = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(sorted_errors[k - 1])


def calibration_window(cal_valid: np.ndarray, tau: np.datetime64, w_days: int) -> tuple[int, int]:
    """Index range [lo, hi) of calibration rows (sorted by valid) with tau - W < valid <= tau."""
    hi = int(np.searchsorted(cal_valid, tau, side="right"))
    lo = int(np.searchsorted(cal_valid, tau - np.timedelta64(w_days, "D"), side="right"))
    return lo, hi


def available_errors(cv: np.ndarray, ca: np.ndarray, ce: np.ndarray, tau: np.datetime64, w_days: int) -> np.ndarray:
    """Errors with tau - W < valid <= tau (frozen window, valid time) and report time <= tau (availability)."""
    lo, hi = calibration_window(cv, tau, w_days)
    m = ca[lo:hi] <= tau
    assert_no_leakage(ca[lo:hi][m], tau)
    return ce[lo:hi][m]


def selected_errors(cv, ca, ce, tau, alpha: float, w_days: int, w_max_days: int = W_MAX_DAYS) -> np.ndarray:
    """The calibration errors AARGUARD actually uses for one target: the W-day set, widened once to W_max if
    ceil((n+1)(1-alpha)) > n (shared with the matched rolling Gaussian comparator)."""
    e = available_errors(cv, ca, ce, tau, w_days)
    if math.ceil((len(e) + 1) * (1 - alpha)) > len(e):
        e = available_errors(cv, ca, ce, tau, w_max_days)
    return e


def assert_no_leakage(cal_valid_used: np.ndarray, tau: np.datetime64) -> None:
    if len(cal_valid_used) and cal_valid_used.max() > tau:
        raise LeakageError(f"calibration error with valid hour {cal_valid_used.max()} > issue time {tau}")


def split_bounds(df: pd.DataFrame, targets: pd.DataFrame, alpha: float, w_days: int, cell_col: str = "cell",
                 w_max_days: int = W_MAX_DAYS, avail_col: str = "t_obs", return_n: bool = False):
    """q_hat for every target row (and, with return_n, the calibration size n actually used, after widening). `df` is the full error history (rows with T observed); `targets` the rows to bound.

    Grouping key: (station, band, cell_col). Returns a Series of q_hat aligned with targets.index.
    """
    out = pd.Series(np.nan, index=targets.index)
    n_used = pd.Series(0, index=targets.index, dtype=int)
    hist = df.dropna(subset=["e", avail_col])
    groups = {k: g.sort_values("valid") for k, g in hist.groupby(["station", "band", cell_col])}
    empty = np.array([], dtype="datetime64[ns]")
    for key, tg in targets.groupby(["station", "band", cell_col]):
        g = groups.get(key)
        cv = _naive(g["valid"].values) if g is not None else empty
        ca = _naive(g[avail_col].values) if g is not None else empty
        ce = g["e"].values if g is not None else np.array([])
        for tau, rows in tg.groupby("issue"):
            t64 = _naive(np.array([tau]))[0]
            e = selected_errors(cv, ca, ce, t64, alpha, w_days, w_max_days)
            q, n = conformal_q(np.sort(e), alpha), len(e)
            out.loc[rows.index] = q
            n_used.loc[rows.index] = n
    return (out, n_used) if return_n else out


def _naive(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return np.array([], dtype="datetime64[ns]")
    s = pd.DatetimeIndex(a)
    if s.tz is not None:
        s = s.tz_convert(None)
    return s.values.astype("datetime64[ns]")


def aci_bounds(df: pd.DataFrame, targets: pd.DataFrame, alpha: float, gamma: float, w_days: int,
               cell_col: str = "cell", w_max_days: int = W_MAX_DAYS, avail_col: str = "t_obs") -> pd.Series:
    """ACI (Gibbs & Candes 2021) per station x band with delayed, batched feedback: each bounded forecast's miss
    (against the bound issued with it) enters as one additive update once its observation report time is <= the current
    issue time. The quantile at level 1 - alpha_t uses the same calibration set as AARGUARD. No guarantee is
    attached to this delayed, batched update; it is evaluated empirically."""
    out = pd.Series(np.nan, index=targets.index)
    hist = df.dropna(subset=["e", avail_col])
    groups = {k: g.sort_values("valid") for k, g in hist.groupby(["station", "band", cell_col])}
    empty = np.array([], dtype="datetime64[ns]")
    for (s, b), tg in targets.groupby(["station", "band"]):
        a_t = alpha
        pending: list[tuple[np.datetime64, float, float]] = []  # (report time, q used at issuance, e) awaiting outcome
        for tau, rows in tg.sort_values("issue").groupby("issue"):
            t64 = _naive(np.array([tau]))[0]
            # feedback: each earlier bounded forecast whose observation was reported by tau, one additive update each
            ready = sorted([p for p in pending if p[0] <= t64], key=lambda p: p[0])
            pending = [p for p in pending if p[0] > t64]
            for _, q_used, e_obs in ready:
                miss = float(e_obs > q_used + 1e-9)            # ties are covered (<=)
                a_t = a_t + gamma * (alpha - miss)
            for cell, rr in rows.groupby(cell_col):
                g = groups.get((s, b, cell))
                cv = _naive(g["valid"].values) if g is not None else empty
                ca = _naive(g[avail_col].values) if g is not None else empty
                ce = g["e"].values if g is not None else np.array([])
                if a_t <= 0:
                    q = math.inf
                elif a_t >= 1:
                    q = -math.inf
                else:
                    q = conformal_q(np.sort(available_errors(cv, ca, ce, t64, w_days)), a_t)
                    if math.isinf(q):
                        q = conformal_q(np.sort(available_errors(cv, ca, ce, t64, w_max_days)), a_t)
                out.loc[rr.index] = q
                # forecasts without an observation never return feedback (counted by aci_feedback_stats)
                ok = rr[avail_col].notna() & rr["e"].notna()
                for v, e_obs in zip(_naive(rr.loc[ok, avail_col].values), rr.loc[ok, "e"].values):
                    pending.append((v, q, e_obs))
    return out
