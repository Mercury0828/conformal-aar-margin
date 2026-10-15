# Exceedance-risk margins for ambient-adjusted line ratings

Code and data for the IEEE TPEC 2027 paper *Exceedance-Risk Margins for Ambient-Adjusted Line Ratings* by Jiachen Shen, Jian Shi, Zirui Tong and Zhu Han.

The package name in the code is `aarguard`; the method label `AARGUARD` in the result tables is the conformal margin of the
paper.

## Contents
| Path | What |
|---|---|
| `src/aarguard/` | data download, alignment of forecasts and observations, IEEE 738 rating map (`rating.py`), conformal margin and its adaptive variant (`bound.py`), comparison margins (`base.py`), bootstrap (`evalstats.py`) |
| `q/` | analyses behind every number in the paper (`run_q.py` coverage and capacity cost, `q4_case.py` heat waves, descriptive scripts) |
| `figs/` | figure and table scripts |
| `data/stations.csv` | the 12 Texas airport stations |
| `data/aligned/` | aligned forecast-observation tables, one parquet file per station: 12 UTC GFS MOS forecast `T_hat`, observed temperature `T`, error `e`, lead, band, cell and the ratings for the conductor cases, January 2018 to May 2025 |
| `data/obs_hourly/` | quality-checked hourly FM-15 temperatures per station (used for the heat-wave calendar) |
| `results/` | the CSVs behind every number in the paper |

## Data sources
* Observations: NOAA Integrated Surface Database (global-hourly, report type FM-15), NCEI Access Data Service.
* Forecasts: NWS GFS MOS (MAV) station guidance from the Iowa Environmental Mesonet MOS archive, 12 UTC cycle, leads 6–72 h.
Both are public archives. `data/aligned/` is derived from them; `python -m aarguard.download` re-downloads the raw files.

## Reproduce
```
pip install -r requirements.txt
export PYTHONPATH=src
export AARGUARD_TEST_OPEN=1          # opens the test period (June 2022 to May 2025)
python q/run_q.py test               # coverage and capacity cost per lead band (writes results/Q*_test.csv)
python q/q4_case.py test             # heat-wave case studies
python figs/table1.py                # Table II of the paper
AARGUARD_TEST_OPEN= python -m pytest tests   # the tests check the closed test-period guard
```
To rebuild `data/aligned/` from the raw archives, download with
`python -m aarguard.download --product ISD --stations <ICAO,...> --years 2018-2025` and
`python -m aarguard.download --product MOS:GFS --span --stations <ICAO,...> --years 2018-2025`, then delete
`data/aligned/` and rerun; raw files go to `data/raw/` (override with `AARGUARD_RAW`).

License: MIT.
