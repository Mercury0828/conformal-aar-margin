"""Checkpoint-and-resume downloader for NOAA ISD observations and IEM MOS archives.

One file per (product, station, year), written atomically (temp file + os.replace). `runs/MANIFEST.csv` records
done / failed items with URL, retrieval time, size and SHA-256. Done items are skipped unless --fresh is given.

Usage:
    python -m aarguard.download --product ISD --stations KAUS,KDFW --years 2018-2025
    python -m aarguard.download --product MOS:GFS --stations KAUS --years 2018-2025
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import hashlib
import os
import sys
import time
from pathlib import Path

import requests

DATA_ROOT = Path(os.environ.get("AARGUARD_RAW", Path(__file__).resolve().parents[2] / "data" / "raw"))
REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "runs" / "MANIFEST.csv"
MANIFEST_FIELDS = ["key", "product", "station", "year", "status", "url", "retrieved_utc", "path", "bytes", "sha256",
                   "note"]

# USAF-WBAN ids from NCEI isd-history.csv (retrieved 2026-10-14), current record per ICAO id.
ISD_IDS = {
    "KDFW": "72259003927", "KIAH": "72243012960", "KAUS": "72254013904", "KSAT": "72253012921",
    "KCRP": "72251012924", "KMAF": "72265023023", "KBRO": "72250012919", "KABI": "72266013962",
    "KELP": "72270023044", "KAMA": "72363023047", "KBPT": "72241012917", "KTXK": "72341813977",
    "KACT": "72256013959", "KSJT": "72263023034", "KLRD": "72252012907", "KHOU": "72244012918",
    "KTYR": "72244813972", "KGGG": "72247003901", "KLBB": "72267023042", "KDAL": "72258013960",
}

IEM_SPACING_S = 12.0  # IEM rate-limits aggressively; the machine shares one IP with other sessions.
NCEI_SPACING_S = 3.0


def set_below_normal_priority() -> None:
    if os.name == "nt":
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)


def read_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        return {row["key"]: row for row in csv.DictReader(f)}


def write_manifest(rows: dict[str, dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow({k: rows[key].get(k, "") for k in MANIFEST_FIELDS})
    os.replace(tmp, MANIFEST)


def target(product: str, station: str, year: int) -> tuple[str, Path]:
    if product == "ISD":
        sid = ISD_IDS[station]
        url = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=global-hourly"
               f"&stations={sid}&startDate={year}-01-01T00:00:00&endDate={year}-12-31T23:59:59"
               "&dataTypes=TMP,REPORT_TYPE&format=csv")
        return url, DATA_ROOT / "noaa" / f"{station}_{year}.csv"
    if product.startswith("MOS:"):
        model = product.split(":", 1)[1]
        url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?"
               f"station={station}&model={model}&sts={year}-01-01T00:00Z&ets={year + 1}-01-01T00:00Z&format=csv")
        return url, DATA_ROOT / "forecasts" / model / f"{station}_{year}.csv"
    raise ValueError(product)


def fetch(url: str, spacing: float, max_tries: int = 6) -> bytes:
    wait = spacing
    for attempt in range(max_tries):
        time.sleep(wait)
        try:
            r = requests.get(url, timeout=600)
        except requests.RequestException as exc:  # network hiccup: back off and retry
            print(f"  network error {exc!r}; retry", flush=True)
            wait = min(600, wait * 3 + 30)
            continue
        if r.status_code == 200:
            return r.content
        print(f"  HTTP {r.status_code}; retry after backoff", flush=True)
        wait = min(900, wait * 3 + 60)
    raise RuntimeError(f"failed after {max_tries} tries: {url}")


def free_gb(path: Path) -> float:
    free = ctypes.c_ulonglong(0)
    if os.name == "nt":
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(str(path.anchor), None, None, ctypes.byref(free))
        return free.value / 1e9
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, help="ISD or MOS:<model>")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--years", required=True, help="e.g. 2018-2025")
    ap.add_argument("--fresh", action="store_true", help="re-download items already marked done")
    ap.add_argument("--span", action="store_true",
                    help="MOS only: one request per station for all pending years, split locally into per-year files")
    args = ap.parse_args(argv)
    set_below_normal_priority()

    y0, y1 = (int(x) for x in args.years.split("-"))
    stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    spacing = NCEI_SPACING_S if args.product == "ISD" else IEM_SPACING_S
    manifest = read_manifest()
    if args.span and args.product.startswith("MOS:"):
        return span_mode(args, stations, y0, y1, manifest)
    for station in stations:
        for year in range(y0, y1 + 1):
            key = f"{args.product}|{station}|{year}"
            if not args.fresh and manifest.get(key, {}).get("status") == "done":
                continue
            if free_gb(DATA_ROOT) < 8.0:
                print("free space below 8 GB on the data drive; pausing (exit)", flush=True)
                return 2
            url, path = target(args.product, station, year)
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{key}: fetching", flush=True)
            row = {"key": key, "product": args.product, "station": station, "year": year, "url": url,
                   "path": str(path)}
            try:
                content = fetch(url, spacing)
                tmp = path.with_suffix(".csv.part")
                tmp.write_bytes(content)
                os.replace(tmp, path)
                row.update(status="done", bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                           retrieved_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            except Exception as exc:  # logged, never dropped
                row.update(status="failed", note=repr(exc)[:300],
                           retrieved_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
            manifest[key] = row
            write_manifest(manifest)
            print(f"{key}: {row['status']} {row.get('bytes', '')}", flush=True)
    return 0


def span_mode(args, stations, y0, y1, manifest) -> int:
    """One IEM request per station covering every pending year; per-year files written atomically."""
    import io
    import pandas as pd
    model = args.product.split(":", 1)[1]
    for station in stations:
        pending = [y for y in range(y0, y1 + 1)
                   if args.fresh or manifest.get(f"{args.product}|{station}|{y}", {}).get("status") != "done"]
        if not pending:
            continue
        if free_gb(DATA_ROOT) < 8.0:
            print("free space below 8 GB on the data drive; pausing (exit)", flush=True)
            return 2
        a, b = min(pending), max(pending)
        url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?"
               f"station={station}&model={model}&sts={a}-01-01T00:00Z&ets={b + 1}-01-01T00:00Z&format=csv")
        print(f"{args.product}|{station}|{a}-{b}: fetching (span)", flush=True)
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            content = fetch(url, IEM_SPACING_S)
            df = pd.read_csv(io.BytesIO(content), dtype=str)
            years = pd.to_datetime(df["runtime"]).dt.year
            for y in pending:
                _, path = target(args.product, station, y)
                path.parent.mkdir(parents=True, exist_ok=True)
                part = df[years == y]
                data = part.to_csv(index=False).encode("utf-8")
                tmp = path.with_suffix(".csv.part")
                tmp.write_bytes(data)
                os.replace(tmp, path)
                manifest[f"{args.product}|{station}|{y}"] = {
                    "key": f"{args.product}|{station}|{y}", "product": args.product, "station": station, "year": y,
                    "status": "done", "url": url, "retrieved_utc": now, "path": str(path), "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "note": f"split from span request {a}-{b}"}
        except Exception as exc:  # logged, never dropped
            for y in pending:
                manifest[f"{args.product}|{station}|{y}"] = {
                    "key": f"{args.product}|{station}|{y}", "product": args.product, "station": station, "year": y,
                    "status": "failed", "url": url, "retrieved_utc": now, "note": repr(exc)[:300]}
        write_manifest(manifest)
        print(f"{args.product}|{station}: {manifest[f'{args.product}|{station}|{pending[0]}']['status']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
