"""Shared data layer: MongoDB handles, a small TTL cache, and period roll-up.

Every chart in the dashboard reads `dc_commodity.ThiTruongDien` filtered by
`dataset` (the source name), exactly as `README_mongodb.md` specifies. The News
tab reads `dc_commodity.TinNganhDien`.

The collection stores the *finest* granularity available, so time aggregation
(daily -> monthly/quarterly/yearly) and derived metrics live here on the server:
the browser receives ready-to-plot rows.
"""

from __future__ import annotations

import os
import time
import tomllib
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from pymongo import MongoClient

DB_NAME = "dc_commodity"
CHARTS_COLLECTION = "ThiTruongDien"
NEWS_COLLECTION = "TinNganhDien"

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

FREQS = ("day", "month", "quarter", "year")


# --------------------------------------------------------------------------
# connection
# --------------------------------------------------------------------------
def load_secret(name: str) -> str:
    """Resolve a secret from the environment, then from .secrets.toml/secrets.toml."""
    if os.getenv(name):
        return os.environ[name]
    for path in (
        PROJECT_ROOT / ".secrets.toml",
        PROJECT_ROOT / "secrets.toml",
        HERE / ".secrets.toml",
    ):
        try:
            if path.exists():
                data = tomllib.loads(path.read_text(encoding="utf-8"))
                if data.get(name):
                    return str(data[name])
        except (OSError, tomllib.TOMLDecodeError):
            continue
    return ""


@lru_cache(maxsize=1)
def get_db():
    uri = load_secret("MONGO_URI")
    if not uri:
        raise RuntimeError("Missing MONGO_URI (set env var or add it to .secrets.toml).")
    return MongoClient(uri)[DB_NAME]


def charts():
    return get_db()[CHARTS_COLLECTION]


# --------------------------------------------------------------------------
# cache — the heavy datasets (107k half-hourly region rows, 23k reservoir rows)
# are re-read on every request otherwise; the pipeline only refreshes daily.
# --------------------------------------------------------------------------
CACHE_TTL = float(os.getenv("DASHBOARD_CACHE_TTL", "300"))
_CACHE: dict[str, tuple[float, object]] = {}


def cached(key: str, build):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    value = build()
    _CACHE[key] = (now, value)
    return value


def clear_cache() -> int:
    n = len(_CACHE)
    _CACHE.clear()
    return n


# --------------------------------------------------------------------------
# period helpers
# --------------------------------------------------------------------------
def period_of(d: datetime, freq: str) -> str:
    """Bucket label for a date: 2026-07-28 -> day/month/quarter/year key."""
    if freq == "day":
        return d.strftime("%Y-%m-%d")
    if freq == "month":
        return d.strftime("%Y-%m")
    if freq == "quarter":
        return f"{d.year}-Q{(d.month - 1) // 3 + 1}"
    return str(d.year)


def period_start(period: str) -> str:
    """ISO date of the first day of a period key — used for sorting/time axes."""
    if len(period) == 4:                      # 2026
        return f"{period}-01-01"
    if "Q" in period:                         # 2026-Q3
        year, q = period.split("-Q")
        return f"{year}-{(int(q) - 1) * 3 + 1:02d}-01"
    if len(period) == 7:                      # 2026-07
        return f"{period}-01"
    return period                             # already a day


def period_sort_key(period: str) -> str:
    return period_start(period)


def year_of(period: str) -> int:
    return int(period[:4])


def within_year(period: str, freq: str) -> str:
    """The part of a period that repeats every year — the x-axis of a year-overlay chart."""
    if freq == "year":
        return period
    if freq == "quarter":
        return period.split("-")[1]           # Q3
    if freq == "month":
        return period[5:7]                    # 07
    return period[5:]                         # 07-28


# --------------------------------------------------------------------------
# roll-up
# --------------------------------------------------------------------------
def _blank(how: str) -> dict:
    return {"sum": 0.0, "n": 0, "max": None, "min": None}


def rollup(rows, freq: str, how: str = "sum", metrics=None):
    """Aggregate `[{date, values{}}]` into periods.

    `how` is `sum` | `avg` | `max` | `last`. Metrics are discovered from the data
    unless `metrics` is given (which also fixes their order).
    """
    buckets: dict[str, dict[str, dict]] = {}
    seen: list[str] = list(metrics or [])
    for row in rows:
        d = row.get("date")
        if not isinstance(d, datetime):
            continue
        key = period_of(d, freq)
        bucket = buckets.setdefault(key, {})
        for metric, raw in (row.get("values") or {}).items():
            if metrics is not None and metric not in metrics:
                continue
            value = _num(raw)
            if value is None:
                continue
            if metric not in seen:
                seen.append(metric)
            acc = bucket.setdefault(metric, _blank(how))
            acc["sum"] += value
            acc["n"] += 1
            acc["max"] = value if acc["max"] is None else max(acc["max"], value)
            acc["min"] = value if acc["min"] is None else min(acc["min"], value)
            acc["last"] = value

    counts: dict[str, int] = {}
    for row in rows:
        d = row.get("date")
        if isinstance(d, datetime):
            key = period_of(d, freq)
            counts[key] = counts.get(key, 0) + 1

    out = []
    for key in sorted(buckets, key=period_sort_key):
        values = {}
        for metric, acc in buckets[key].items():
            if how == "avg":
                values[metric] = acc["sum"] / acc["n"] if acc["n"] else None
            elif how == "max":
                values[metric] = acc["max"]
            elif how == "min":
                values[metric] = acc["min"]
            elif how == "last":
                values[metric] = acc.get("last")
            else:
                values[metric] = acc["sum"]
        # `n` = source documents behind the bucket. A month that only has a few
        # daily rows (the scrapes have gaps) sums to a misleadingly small total,
        # so the client can flag partial periods.
        out.append({"period": key, "date": period_start(key), "values": values, "n": counts.get(key, 0)})
    return out, seen


def _num(x):
    """Coerce a stored value to float; None for blanks and non-numerics."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.replace(",", "").strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


num = _num


# --------------------------------------------------------------------------
# generic reader
# --------------------------------------------------------------------------
def read(dataset: str, objects=None, date_from=None, date_to=None, projection=None):
    """Raw documents of one dataset, oldest first."""
    query: dict = {"dataset": dataset}
    if objects:
        query["object"] = {"$in": list(objects)}
    span: dict = {}
    if date_from:
        span["$gte"] = date_from
    if date_to:
        span["$lte"] = date_to
    if span:
        query["date"] = span
    fields = projection or {"object": 1, "date": 1, "values": 1, "units": 1}
    return list(charts().find(query, fields).sort("date", 1))


def units_of(dataset: str) -> dict:
    """Unit map of a dataset, taken from its most recent document."""
    doc = charts().find_one({"dataset": dataset, "units": {"$ne": {}}}, {"units": 1}, sort=[("date", -1)])
    return (doc or {}).get("units") or {}


def parse_date(text: str):
    """`YYYY-MM-DD` -> datetime, or None when absent/invalid."""
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None
