"""Per-chart computations — one function per chart in `README_mongodb.md`.

Each function returns rows the browser can plot directly. Everything the guide
lists as "the frontend is responsible for" (time aggregation, YoY, averages,
CGM = SMP + CAN) happens here so the client stays a thin renderer.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from datetime import datetime

from .data import (
    cached,
    charts,
    num,
    period_of,
    period_sort_key,
    period_start,
    read,
    rollup,
    units_of,
    PROJECT_ROOT,
)

# ==========================================================================
# Tab 2 — Weather
# ==========================================================================
ENSO_METRICS = ["La Nina prob", "Neutral prob", "El Nino prob"]


def enso_probability() -> dict:
    """`fetch_enso_probability_monthly` — probability split per forecast issue month.

    The guide offers a "view by issue date" option, but `upload_csv.py` does not
    split the `season` horizon column into `values` (§5.3), so each document is a
    single issue month and the stacked bar runs over issue dates.
    """
    def build():
        rows = []
        for doc in read("fetch_enso_probability_monthly"):
            values = doc.get("values") or {}
            rows.append({
                "date": doc["date"].strftime("%Y-%m-%d"),
                "values": {m: num(values.get(m)) for m in ENSO_METRICS},
            })
        rows.sort(key=lambda r: r["date"])
        return {"metrics": ENSO_METRICS, "rows": rows, "unit": "%"}

    return cached("weather:enso_prob", build)


def enso_condition(date_from=None, date_to=None) -> dict:
    """`fetch_hydrology_current_monthly` — ONI, coloured El Niño / La Niña / Neutral."""
    def build():
        rows = []
        for doc in read("fetch_hydrology_current_monthly"):
            oni = num((doc.get("values") or {}).get("oni_value"))
            if oni is None:
                continue
            phase = "el_nino" if oni > 0.5 else "la_nina" if oni < -0.5 else "neutral"
            rows.append({"date": doc["date"].strftime("%Y-%m-%d"), "oni": oni, "phase": phase})
        rows.sort(key=lambda r: r["date"])
        return rows

    rows = cached("weather:enso_cond", build)
    lo = date_from.strftime("%Y-%m-%d") if date_from else None
    hi = date_to.strftime("%Y-%m-%d") if date_to else None
    if lo or hi:
        rows = [r for r in rows if (not lo or r["date"] >= lo) and (not hi or r["date"] <= hi)]
    return {"rows": rows, "unit": "ONI"}


CITIES = {"Ha_Noi": "Hà Nội", "Da_Nang": "Đà Nẵng", "Ho_Chi_Minh": "Sài Gòn"}


def temperature(freq: str = "month") -> dict:
    """`fetch_weather_temperature_monthly` — mean temperature, averaged per period.

    Daily granularity in the collection; one series per city, averaged into the
    requested bucket. The client overlays years on a common x-axis.
    """
    def build_daily():
        per_city: dict[str, list] = defaultdict(list)
        for doc in read("fetch_weather_temperature_monthly"):
            value = num((doc.get("values") or {}).get("temp_mean"))
            if value is None:
                continue
            per_city[doc["object"]].append({"date": doc["date"], "values": {"temp": value}})
        return per_city

    per_city = cached("weather:temp_daily", build_daily)
    series = {}
    for city, rows in per_city.items():
        buckets, _ = rollup(rows, freq, how="avg")
        series[CITIES.get(city, city)] = [
            {"period": b["period"], "date": b["date"], "value": b["values"].get("temp")}
            for b in buckets
        ]
    return {"cities": [CITIES.get(c, c) for c in CITIES if CITIES.get(c, c) in series],
            "series": series, "unit": "°C", "freq": freq}


# ==========================================================================
# Tab 3 — Cost
# ==========================================================================
def coal_cost() -> dict:
    """`manual_coal_cost_monthly` — plant coal cost, VND/ton.

    Metric names in the collection are `Cost for Vinh Tan` / `Cost for Mong Duong`
    (the guide writes them as "Coal Cost for …").
    """
    def build():
        metrics: list[str] = []
        rows = []
        for doc in read("manual_coal_cost_monthly"):
            values = {k: num(v) for k, v in (doc.get("values") or {}).items()}
            for key in values:
                if key not in metrics:
                    metrics.append(key)
            rows.append({"date": doc["date"].strftime("%Y-%m-%d"), "values": values})
        rows.sort(key=lambda r: r["date"])
        return {"metrics": metrics, "rows": rows, "unit": "VND/tấn",
                "units": units_of("manual_coal_cost_monthly")}

    return cached("cost:coal", build)


def gas_cost() -> dict:
    """`manual_gas_cost_monthly` — `object` = plant, `values["Gas Cost"]` in USD/MMBTU."""
    def build():
        by_date: dict[str, dict] = {}
        plants: list[str] = []
        for doc in read("manual_gas_cost_monthly"):
            value = num((doc.get("values") or {}).get("Gas Cost"))
            if value is None:
                continue
            plant = doc["object"]
            if plant not in plants:
                plants.append(plant)
            by_date.setdefault(doc["date"].strftime("%Y-%m-%d"), {})[plant] = value
        rows = [{"date": d, "values": v} for d, v in sorted(by_date.items())]
        return {"metrics": plants, "rows": rows, "unit": "USD/MMBTU"}

    return cached("cost:gas", build)


def _reservoir_regions() -> dict[str, str]:
    """Reservoir -> region.

    The reservoir documents carry no region (§5.2), so the mapping comes from
    `fact_water_list` in MongoDB. That table only lists company-owned lakes, so
    the reservoirs it misses are topped up from the region column of the source
    CSV when the file is available; anything still unmapped is grouped separately.
    """
    def build():
        mapping: dict[str, str] = {}
        for doc in charts().find({"dataset": "fact_water_list"}, {"lake": 1, "region": 1}):
            lake, region = doc.get("lake"), doc.get("region")
            if lake and region:
                mapping.setdefault(lake, region)
        path = PROJECT_ROOT / "data" / "industries_data" / "fetch_hydro_reservoir_monthly.csv"
        try:
            raw = path.read_bytes().decode("utf-8-sig", "replace")
        except OSError:
            return mapping
        for row in csv.DictReader(io.StringIO(raw)):
            name, region = (row.get("reservoir_name") or "").strip(), (row.get("region") or "").strip()
            if name and region:
                mapping.setdefault(name, region)
        return mapping

    return cached("cost:region_map", build)


UNMAPPED_REGION = "Chưa gán vùng"
_CACHE_NOTE: dict[str, int] = {}


# A reservoir's level only moves between its dead level and its full level, so a
# reading several times the lake's own median is not a real level — the upstream
# scrape occasionally mis-parses a thousands separator (Đại Ninh reads 217 853 m
# where the lake sits near 868 m). Left in, one such day multiplies a monthly mean
# by 250× and the YoY chart becomes a single spike. Drop those readings.
OUTLIER_FACTOR = 3.0


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _reservoir_period_avg(freq: str) -> dict[tuple[str, str], float]:
    """Average flood level per (reservoir, period) — the base of the YoY chart."""
    def build_monthly():
        readings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        cursor = charts().find(
            {"dataset": "fetch_hydro_reservoir_monthly"},
            {"object": 1, "date": 1, "values.flood_level": 1},
        )
        for doc in cursor:
            level = num((doc.get("values") or {}).get("flood_level"))
            if level is None or level <= 0 or not isinstance(doc.get("date"), datetime):
                continue
            readings[doc["object"]].append((doc["date"].strftime("%Y-%m"), level))

        acc: dict[tuple[str, str], list[float]] = defaultdict(list)
        dropped = 0
        for name, points in readings.items():
            mid = _median([v for _, v in points])
            lo, hi = mid / OUTLIER_FACTOR, mid * OUTLIER_FACTOR
            for month, value in points:
                if lo <= value <= hi:
                    acc[(name, month)].append(value)
                else:
                    dropped += 1
        _CACHE_NOTE["reservoir_dropped"] = dropped
        return {k: sum(v) / len(v) for k, v in acc.items()}

    monthly = cached("cost:reservoir_monthly", build_monthly)
    if freq == "month":
        return monthly
    # Roll the monthly means up — an unweighted mean of months, matching step 1
    # of the guide's computation chain ("monthly average per reservoir" first).
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (name, month), value in monthly.items():
        year, mm = month.split("-")
        period = year if freq == "year" else f"{year}-Q{(int(mm) - 1) // 3 + 1}"
        acc[(name, period)].append(value)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _previous_period(period: str, freq: str) -> str:
    """The same period one year earlier."""
    if freq == "year":
        return str(int(period) - 1)
    head, tail = period.split("-")
    return f"{int(head) - 1}-{tail}"


def reservoir_yoy(freq: str = "month") -> dict:
    """Region-level YoY growth of reservoir flood level.

    Chain from the guide: period average per reservoir -> YoY growth per
    reservoir -> average those growth rates into a region.
    """
    if freq not in ("month", "quarter", "year"):
        freq = "month"

    def build():
        averages = _reservoir_period_avg(freq)
        regions = _reservoir_regions()
        growth: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for (name, period), value in averages.items():
            previous = averages.get((name, _previous_period(period, freq)))
            if previous in (None, 0):
                continue
            region = regions.get(name, UNMAPPED_REGION)
            growth[period][region].append((value - previous) / previous * 100.0)

        region_names = sorted({r for per in growth.values() for r in per}, key=lambda r: (r == UNMAPPED_REGION, r))
        rows = []
        for period in sorted(growth, key=period_sort_key):
            rows.append({
                "period": period,
                "date": period_start(period),
                "values": {
                    region: (sum(v) / len(v) if (v := growth[period].get(region)) else None)
                    for region in region_names
                },
            })
        dropped = _CACHE_NOTE.get("reservoir_dropped", 0)
        return {
            "metrics": region_names, "rows": rows, "unit": "%", "freq": freq,
            "dropped_readings": dropped,
            "note": (f"Đã loại {dropped} số đo mực nước sai lệch bất thường "
                     f"(lệch quá {OUTLIER_FACTOR:g} lần trung vị của chính hồ đó)."
                     if dropped else ""),
        }

    return cached(f"cost:reservoir_yoy:{freq}", build)


# ==========================================================================
# Tab 4 — Price
# ==========================================================================
SMP_REGIONS = {
    "giaBienMB": "Miền Bắc",
    "giaBienMT": "Miền Trung",
    "giaBienMN": "Miền Nam",
    "giaBienHT": "Toàn hệ thống",
}


def _smp_daily() -> list[dict]:
    """Daily average SMP per region.

    `fetch_price_smp_monthly` nests a half-hourly map under `values` (§5.1). Slots
    that have not settled yet are stored as `0.0`; they are excluded so a partial
    day is not dragged towards zero.
    """
    def build():
        rows = []
        for doc in read("fetch_price_smp_monthly"):
            totals = defaultdict(float)
            counts = defaultdict(int)
            for slot in (doc.get("values") or {}).values():
                if not isinstance(slot, dict):
                    continue
                for key, raw in slot.items():
                    value = num(raw)
                    if value:            # 0.0 == not settled -> skip
                        totals[key] += value
                        counts[key] += 1
            values = {k: totals[k] / counts[k] for k in totals if counts[k]}
            if values:
                rows.append({"date": doc["date"], "values": values})
        return rows

    return cached("price:smp_daily", build)


def _can_by_year() -> dict[int, float]:
    """`manual_price_can_annually` — CAN price keyed by year."""
    def build():
        out = {}
        for doc in read("manual_price_can_annually"):
            value = num((doc.get("values") or {}).get("CAN price"))
            if value is not None:
                out[doc["date"].year] = value
        return out

    return cached("price:can_year", build)


def cgm_price(freq: str = "month") -> dict:
    """CGM = daily SMP average + that year's CAN, averaged into the period.

    One series set per region; the client draws four charts and overlays years.
    """
    def build():
        can = _can_by_year()
        smp_rows, cgm_rows = [], []
        for row in _smp_daily():
            year = row["date"].year
            add = can.get(year)
            smp_rows.append(row)
            if add is None:
                continue
            cgm_rows.append({
                "date": row["date"],
                "values": {k: v + add for k, v in row["values"].items()},
            })
        smp, _ = rollup(smp_rows, freq, how="avg")
        cgm, _ = rollup(cgm_rows, freq, how="avg")
        regions = list(SMP_REGIONS.items())
        return {
            "regions": [{"key": k, "label": v} for k, v in regions],
            "smp": smp,
            "cgm": cgm,
            "can": {str(y): v for y, v in sorted(can.items())},
            "unit": "VND/kWh",
            "freq": freq,
        }

    return cached(f"price:cgm:{freq}", build)


def can_price() -> dict:
    """`manual_price_can_monthly` does not exist yet (§5.5) — the annual file stands in."""
    def build():
        rows = [
            {"date": doc["date"].strftime("%Y-%m-%d"),
             "period": str(doc["date"].year),
             "value": num((doc.get("values") or {}).get("CAN price"))}
            for doc in read("manual_price_can_annually")
        ]
        rows = [r for r in rows if r["value"] is not None]
        rows.sort(key=lambda r: r["date"])
        return {
            "rows": rows,
            "unit": "VND/kWh",
            "note": "Nguồn tháng (manual_price_can_monthly) chưa có — đang dùng file năm "
                    "manual_price_can_annually.",
        }

    return cached("price:can", build)


def retail_price() -> dict:
    """`manual_price_retail_monthly` — the regulated retail price steps."""
    def build():
        rows = [
            {"date": doc["date"].strftime("%Y-%m-%d"),
             "value": num((doc.get("values") or {}).get("Retail price"))}
            for doc in read("manual_price_retail_monthly")
        ]
        rows = [r for r in rows if r["value"] is not None]
        rows.sort(key=lambda r: r["date"])
        return {"rows": rows, "unit": "VND/kWh"}

    return cached("price:retail", build)


# ==========================================================================
# Tab 5 — Volume
# ==========================================================================
VOLUME_TOTAL_METRIC = "generation_thuong_pham_mkWh"
VOLUME_BREAKDOWN = {
    "thuy_dien_mkWh": "Thủy điện",
    "nhiet_dien_than_mkWh": "Nhiệt điện than",
    "tuabin_khi_mkWh": "Tuabin khí",
    "nhiet_dien_dau_mkWh": "Nhiệt điện dầu",
    "dien_gio_mkWh": "Điện gió",
    "dmt_trang_trai_mkWh": "ĐMT trang trại",
    "dmt_mai_thuong_pham_mkWh": "ĐMT mái nhà",
    "nhap_khau_mkWh": "Nhập khẩu",
    "khac_mkWh": "Khác",
}
DEMAND_METRIC = "max_power_dau_cuc_MW"
SUPPLY_OBJECT = "Quốc gia + ĐMT mái nhà (ước tính đầu cực)"
SUPPLY_METRIC = "mobilized_high_season"


def _volume_source_rows() -> list[dict]:
    def build():
        return [
            {"date": doc["date"], "values": {k: num(v) for k, v in (doc.get("values") or {}).items()}}
            for doc in read("fetch_volume_source_monthly")
        ]

    return cached("volume:source_rows", build)


def volume_total(freq: str = "month") -> dict:
    """Commercial generation, summed into the period."""
    rows, _ = rollup(_volume_source_rows(), freq, how="sum", metrics=[VOLUME_TOTAL_METRIC])
    return {
        "metric": VOLUME_TOTAL_METRIC,
        "rows": [{"period": r["period"], "date": r["date"], "n": r["n"],
                  "value": r["values"].get(VOLUME_TOTAL_METRIC)}
                 for r in rows],
        "unit": "triệu kWh",
        "freq": freq,
    }


def volume_breakdown(freq: str = "month") -> dict:
    """Generation split by power source, summed into the period."""
    keys = list(VOLUME_BREAKDOWN)
    rows, _ = rollup(_volume_source_rows(), freq, how="sum", metrics=keys)
    return {
        "metrics": [VOLUME_BREAKDOWN[k] for k in keys],
        "rows": [
            {"period": r["period"], "date": r["date"], "n": r["n"],
             "values": {VOLUME_BREAKDOWN[k]: r["values"].get(k) for k in keys}}
            for r in rows
        ],
        "unit": "triệu kWh",
        "freq": freq,
    }


def capacity_mismatch(freq: str = "month") -> dict:
    """Peak available capacity (supply) against peak demand — both period maxima."""
    def build():
        supply_docs = read("fetch_capacity_mobilized_monthly", objects=[SUPPLY_OBJECT])
        return [
            {"date": doc["date"], "values": {"supply": num((doc.get("values") or {}).get(SUPPLY_METRIC))}}
            for doc in supply_docs
        ]

    supply_rows = cached("volume:supply_rows", build)
    supply, _ = rollup(supply_rows, freq, how="max", metrics=["supply"])
    demand, _ = rollup(_volume_source_rows(), freq, how="max", metrics=[DEMAND_METRIC])

    merged: dict[str, dict] = {}
    for row in supply:
        merged.setdefault(row["period"], {"period": row["period"], "date": row["date"]})["supply"] = \
            row["values"].get("supply")
    for row in demand:
        merged.setdefault(row["period"], {"period": row["period"], "date": row["date"]})["demand"] = \
            row["values"].get(DEMAND_METRIC)
    rows = sorted(merged.values(), key=lambda r: period_sort_key(r["period"]))
    for row in rows:
        supply_value, demand_value = row.get("supply"), row.get("demand")
        row["gap"] = (supply_value - demand_value) if None not in (supply_value, demand_value) else None
    return {"rows": rows, "unit": "MW", "freq": freq,
            "supply_object": SUPPLY_OBJECT, "supply_metric": SUPPLY_METRIC}


NATIONAL_OBJECT = "Toàn quốc"


def capacity_installed() -> dict:
    """Installed capacity by power type on the latest day — the pie chart."""
    def build():
        latest = charts().find_one(
            {"dataset": "fetch_capacity_installed_monthly"}, {"date": 1}, sort=[("date", -1)]
        )
        if not latest:
            return {"rows": [], "total": None, "date": None, "unit": "MW"}
        docs = read("fetch_capacity_installed_monthly", date_from=latest["date"], date_to=latest["date"])
        rows, total = [], None
        for doc in docs:
            value = num((doc.get("values") or {}).get("installed_capacity"))
            if value is None:
                continue
            if doc["object"] == NATIONAL_OBJECT:      # the national total, not a slice
                total = value
                continue
            rows.append({"type": doc["object"], "value": value})
        rows.sort(key=lambda r: -r["value"])
        if total is None:
            total = sum(r["value"] for r in rows)
        return {"rows": rows, "total": total,
                "date": latest["date"].strftime("%Y-%m-%d"), "unit": "MW"}

    return cached("volume:capacity_installed", build)


REGION_KEYS = {"congSuatMB": "Miền Bắc", "congSuatMT": "Miền Trung", "congSuatMN": "Miền Nam"}
SYSTEM_KEY = "congSuatHT"


def capacity_region(freq: str = "month") -> dict:
    """Average load per region — half-hourly documents averaged into the period."""
    def build_daily():
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        cursor = charts().find(
            {"dataset": "fetch_volume_region_monthly"}, {"date": 1, "values": 1}
        )
        for doc in cursor:
            if not isinstance(doc.get("date"), datetime):
                continue
            day = doc["date"].strftime("%Y-%m-%d")
            for key, raw in (doc.get("values") or {}).items():
                value = num(raw)
                if value is not None:
                    acc[day][key].append(value)
        rows = []
        for day in sorted(acc):
            rows.append({
                "date": datetime.strptime(day, "%Y-%m-%d"),
                "values": {k: sum(v) / len(v) for k, v in acc[day].items() if v},
            })
        return rows

    daily = cached("volume:region_daily", build_daily)
    keys = list(REGION_KEYS) + [SYSTEM_KEY]
    buckets, _ = rollup(daily, freq, how="avg", metrics=keys)
    return {
        "metrics": [REGION_KEYS[k] for k in REGION_KEYS],
        "system_label": "Toàn hệ thống",
        "rows": [
            {"period": b["period"], "date": b["date"],
             "values": {REGION_KEYS[k]: b["values"].get(k) for k in REGION_KEYS},
             "system": b["values"].get(SYSTEM_KEY)}
            for b in buckets
        ],
        "unit": "MW",
        "freq": freq,
    }


# ==========================================================================
# Tab 6 — Company
# ==========================================================================
# `unit` = what a non-segment `object` names in that dataset: a generation source
# (the quarterly files) or a power plant (PGV/POW, monthly) — the split the
# README's Tab 6 table makes.
COMPANIES = {
    "GEG": {"dataset": "company_geg_quarterly", "cadence": "quarterly", "unit": "source", "name": "Điện Gia Lai"},
    "HDG": {"dataset": "company_hdg_quarterly", "cadence": "quarterly", "unit": "source", "name": "Tập đoàn Hà Đô"},
    "HND": {"dataset": "company_hnd_quarterly", "cadence": "quarterly", "unit": "source", "name": "Nhiệt điện Hải Phòng"},
    "PPC": {"dataset": "company_ppc_quarterly", "cadence": "quarterly", "unit": "source", "name": "Nhiệt điện Phả Lại"},
    "QTP": {"dataset": "company_qtp_quarterly", "cadence": "quarterly", "unit": "source", "name": "Nhiệt điện Quảng Ninh"},
    "PC1": {"dataset": "company_pc1_quarterly", "cadence": "quarterly", "unit": "source", "name": "Tập đoàn PC1"},
    "REE": {"dataset": "company_ree_quarterly", "cadence": "quarterly", "unit": "source", "name": "Cơ Điện Lạnh REE"},
    "PGV": {"dataset": "company_pgv_monthly", "cadence": "monthly", "unit": "plant", "name": "EVNGENCO3"},
    "POW": {"dataset": "company_pow_monthly", "cadence": "monthly", "unit": "plant", "name": "PV Power"},
}

# §5.7 — an `object` is either a business segment (`net revenue`/`gross profit`/
# `NPAT`) or an operating unit (`Volume`/`Mobilized`/`Contracted`/`Revenue`/`ASP`).
# Classify by the metrics present, not by name: REE's generation sources also
# carry `NPAT`, and GEG's `Others` carries only `Revenue` and belongs with the
# generation series, not on its own.
SEGMENT_METRICS = {"net revenue", "gross profit", "NPAT"}
OPERATING_METRICS = {"Volume", "Mobilized", "Contracted"}


def company(ticker: str) -> dict:
    """Every series of one company, grouped `object -> metric -> [{date, value}]`."""
    ticker = ticker.upper()
    meta = COMPANIES.get(ticker)
    if not meta:
        return {}

    def build():
        dataset = meta["dataset"]
        series: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        periods: set[str] = set()
        for doc in read(dataset):
            date = doc["date"]
            period = (period_of(date, "quarter") if meta["cadence"] == "quarterly"
                      else period_of(date, "month"))
            periods.add(period)
            for metric, raw in (doc.get("values") or {}).items():
                value = num(raw)
                if value is None:
                    continue
                series[doc["object"]][metric].append(
                    {"period": period, "date": date.strftime("%Y-%m-%d"), "value": value}
                )
        for metrics in series.values():
            for rows in metrics.values():
                rows.sort(key=lambda r: r["date"])

        objects = sorted(series)
        # A segment reports segment P&L and nothing operational; everything else
        # (minus the company-total object) is an operating unit.
        segments = [
            o for o in objects
            if o != ticker
            and SEGMENT_METRICS & set(series[o])
            and not OPERATING_METRICS & set(series[o])
        ]
        operating = [o for o in objects if o != ticker and o not in segments]
        is_plant = meta["unit"] == "plant"
        return {
            "ticker": ticker,
            "name": meta["name"],
            "dataset": dataset,
            "cadence": meta["cadence"],
            "unit": meta["unit"],
            "periods": sorted(periods, key=period_sort_key),
            "objects": objects,
            "generation": [] if is_plant else operating,
            "segments": segments,
            "plants": operating if is_plant else [],
            "total_object": ticker if ticker in series else None,
            "series": {o: dict(m) for o, m in series.items()},
            "units": units_of(dataset),
        }

    return cached(f"company:{ticker}", build)


# --- reservoirs owned by a company ----------------------------------------
OUT_OF_FILE = "(ngoài file)"


def _water_rows() -> list[dict]:
    def build():
        rows = []
        for doc in charts().find(
            {"dataset": "fact_water_list"},
            {"ticker": 1, "lake": 1, "plants": 1, "region": 1, "values.mw": 1},
        ):
            rows.append({
                "ticker": (doc.get("ticker") or "").upper(),
                "lake": doc.get("lake") or "",
                "plants": doc.get("plants") or "",
                "region": doc.get("region"),
                "mw": num((doc.get("values") or {}).get("mw")) or 0.0,
            })
        return rows

    return cached("company:water_rows", build)


def _reservoir_ytd(lake: str, cutoff_month: int, cutoff_day: int) -> list[dict]:
    """Year-to-date average flood level per year, all years cut at the same day."""
    def build():
        points = []
        cursor = charts().find(
            {"dataset": "fetch_hydro_reservoir_monthly", "object": lake},
            {"date": 1, "values.flood_level": 1},
        )
        for doc in cursor:
            date = doc.get("date")
            level = num((doc.get("values") or {}).get("flood_level"))
            if level is None or level <= 0 or not isinstance(date, datetime):
                continue
            points.append((date, level))
        if not points:
            return []
        # Same outlier guard as the YoY chart — a mis-parsed reading would
        # otherwise lift a whole year's bar.
        mid = _median([v for _, v in points])
        lo, hi = mid / OUTLIER_FACTOR, mid * OUTLIER_FACTOR
        acc: dict[int, list[float]] = defaultdict(list)
        for date, level in points:
            if (date.month, date.day) > (cutoff_month, cutoff_day) or not (lo <= level <= hi):
                continue
            acc[date.year].append(level)
        return [{"year": y, "value": sum(v) / len(v), "readings": len(v)}
                for y, v in sorted(acc.items())]

    return cached(f"company:ytd:{lake}:{cutoff_month:02d}-{cutoff_day:02d}", build)


def _expand_lake(lake: str) -> list[str]:
    """`"Vĩnh Sơn A/B/C"` -> the three reservoir names the flood-level feed stores.

    PGV's `fact_water_list` row collapses the three Vĩnh Sơn reservoirs into one
    entry; `fetch_hydro_reservoir_monthly` keeps them apart.
    """
    if "/" not in lake:
        return [lake]
    base, _, suffixes = lake.rpartition(" ")
    if not base:
        return [lake]
    return [f"{base} {s.strip()}" for s in suffixes.split("/") if s.strip()]


def company_reservoirs(ticker: str, today: datetime | None = None) -> dict:
    """Per-lake YTD-average flood level, with the MW each lake influences.

    Company total MW is the sum of **distinct plant groups** (lakes feeding the
    same group count once) and includes the `(ngoài file)` capacity, so a lake's
    share reads "this reservoir affects X% of the company's hydro capacity".
    Those shares overlap and do not sum to 100%.
    """
    ticker = ticker.upper()
    today = today or datetime.now()
    rows = [r for r in _water_rows() if r["ticker"] == ticker]
    if not rows:
        return {"ticker": ticker, "lakes": [], "total_mw": 0.0}

    groups: dict[str, float] = {}
    for row in rows:
        key = row["plants"] or f"__{row['lake']}"
        groups.setdefault(key, row["mw"])
    total_mw = sum(groups.values())

    charted = charts().distinct("object", {"dataset": "fetch_hydro_reservoir_monthly"})
    lakes = []
    for row in sorted(rows, key=lambda r: -r["mw"]):
        if row["lake"] == OUT_OF_FILE:
            continue
        for name in _expand_lake(row["lake"]):
            if name not in charted:
                continue
            lakes.append({
                "lake": name,
                "region": row["region"],
                "mw": row["mw"],
                "share": (row["mw"] / total_mw * 100.0) if total_mw else None,
                "plants": row["plants"],
                "years": _reservoir_ytd(name, today.month, today.day),
            })
    return {
        "ticker": ticker,
        "lakes": lakes,
        "total_mw": total_mw,
        "cutoff": today.strftime("%d/%m"),
        "unit": "m",
        "note": "MW và % là mức ảnh hưởng của từng hồ, không cộng dồn "
                "(nhiều hồ cùng cấp nước cho một cụm nhà máy).",
    }


# --- the per-company news key map (README §Tab 6) --------------------------
COMPANY_NEWS_KEYS = {
    "GEG": {"tickers": ["GEG"], "search": ["GEG", "Điện Gia Lai GEG"]},
    "HDG": {"tickers": ["HDG"], "search": ["HDG", "Tập đoàn Hà Đô"]},
    "PC1": {"tickers": ["PC1"], "search": ["PC1"]},
    "PGV": {"tickers": ["PGV"], "search": ["PGV", "EVNGenco3 PGV", "EVNGENCO3"]},
    "POW": {"tickers": ["POW", "NT2"],
            "search": ["POW", "PV POW", "Điện lực Dầu khí Nhơn Trạch 2",
                       "Nhơn Trạch 3", "Nhơn Trạch 4"]},
    "REE": {"tickers": ["REE"], "search": ["REE"]},
    "HND": {"tickers": ["HND"], "search": ["HND"]},
    "PPC": {"tickers": ["PPC"], "search": ["PPC"]},
    "QTP": {"tickers": ["QTP"], "search": ["QTP"]},
}
