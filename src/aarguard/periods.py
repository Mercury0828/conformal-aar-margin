"""Frozen periods and the test-period guard.

The test period is opened explicitly (AARGUARD_TEST_OPEN=1); it stayed closed until the analysis was frozen. Every loader that returns values calls
`drop_test` so that no observation, forecast, error or rating with a valid hour in the test period leaves the module.
Availability counts in the test period use `count_only_test` and never see values.
"""
from __future__ import annotations

import os

import pandas as pd

HISTORY_START = pd.Timestamp("2018-01-01 00:00", tz="UTC")
VAL_START = pd.Timestamp("2021-06-01 00:00", tz="UTC")
VAL_END = pd.Timestamp("2022-06-01 00:00", tz="UTC")  # exclusive
TEST_START = VAL_END
TEST_END = pd.Timestamp("2025-06-01 00:00", tz="UTC")  # exclusive

TEST_OPEN = os.environ.get("AARGUARD_TEST_OPEN") == "1"


def drop_test(df: pd.DataFrame, time_cols: tuple[str, ...] = ("valid",)) -> pd.DataFrame:
    """Remove every row whose timestamps reach the test period (unless the test period is formally open)."""
    if TEST_OPEN:
        return df
    keep = pd.Series(True, index=df.index)
    for c in time_cols:
        keep &= df[c] < TEST_START
    return df.loc[keep]


def in_validation(valid: pd.Series) -> pd.Series:
    return (valid >= VAL_START) & (valid < VAL_END)
