"""M-eval helpers: day-block bootstrap (blocks = valid calendar days, UTC; all stations' rows of a day move together).

Conventions: 1,000 resamples, seed 20261014, percentile 95 % CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_BOOT = 1000
SEED = 20261014


def day_blocks(valid: pd.Series) -> np.ndarray:
    return valid.dt.floor("D").values


def bootstrap(df: pd.DataFrame, stat, day_col: str = "day", n_boot: int = N_BOOT, seed: int = SEED):
    """Point estimate, 95 % percentile CI and bootstrap SE of stat(df) under day-block resampling."""
    point = stat(df)
    days = df[day_col].values
    uniq, inv = np.unique(days, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[k]:ends[k]] for k in pick])
        vals[b] = stat(df.iloc[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), float(np.nanstd(vals, ddof=1))


def ratio_bootstrap(num: pd.Series, den: pd.Series, day: pd.Series, n_boot: int = N_BOOT, seed: int = SEED):
    """Fast day-block bootstrap for ratio-of-sums statistics (rates, coverage, CC)."""
    g = pd.DataFrame({"n": num.values, "d": den.values, "day": day.values}).groupby("day").sum()
    n, d = g["n"].values, g["d"].values
    point = n.sum() / d.sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (n_boot, len(g)))
    vals = n[pick].sum(1) / d[pick].sum(1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), float(vals.std(ddof=1))


def block_ids(valid: pd.Series, kind: str, start: pd.Timestamp) -> pd.Series:
    """'day' = UTC calendar day; 'week' = non-overlapping 7-day UTC blocks anchored at the evaluation-period start
    (final shorter block retained)."""
    if kind == "day":
        return valid.dt.floor("D")
    return ((valid - start) // pd.Timedelta(days=7)).astype(int)


def ratio_bootstrap_full(num: pd.Series, den: pd.Series, block: pd.Series, n_boot: int = N_BOOT, seed: int = SEED):
    """Point, two-sided 95 % percentile CI, one-sided 95 % lower bound (5th percentile) and SE for a ratio of sums."""
    g = pd.DataFrame({"n": num.values, "d": den.values, "b": block.values}).groupby("b").sum()
    n, d = g["n"].values, g["d"].values
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (n_boot, len(g)))
    vals = n[pick].sum(1) / d[pick].sum(1)
    lo, hi, p5 = np.percentile(vals, [2.5, 97.5, 5.0])
    return float(n.sum() / d.sum()), float(lo), float(hi), float(p5), float(vals.std(ddof=1))


def paired_ratio_difference(num_a, den_a, num_b, den_b, block, n_boot: int = N_BOOT, seed: int = SEED):
    """Difference of two ratios of sums (A - B) with identical block draws for both methods."""
    g = pd.DataFrame({"na": np.asarray(num_a), "da": np.asarray(den_a), "nb": np.asarray(num_b),
                      "db": np.asarray(den_b), "b": np.asarray(block)}).groupby("b").sum()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(g), (n_boot, len(g)))
    a = g["na"].values[pick].sum(1) / g["da"].values[pick].sum(1)
    b = g["nb"].values[pick].sum(1) / g["db"].values[pick].sum(1)
    point = g["na"].sum() / g["da"].sum() - g["nb"].sum() / g["db"].sum()
    lo, hi = np.percentile(a - b, [2.5, 97.5])
    return float(point), float(lo), float(hi)
