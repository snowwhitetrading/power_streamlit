"""Tab 1 reference tables — read as whole rows.

`README_mongodb.md` §5.4 flags `fact_pdp8_capex`, `fact_pdp8_plant` and
`fact_price_framework` as **dimensional tables**: their key columns are text and
they have no time axis, so the generic `object`/`values` normalisation in
`upload_csv.py` is lossy. In the live collection each of them survives as a
single document holding one row's numbers — not enough to draw the charts.
`fact_pdp8_capacity` loses a column the same way: its `Thấp` and `Cao` 2030
scenarios share a date and collapse onto one document.

So Tab 1 reads these four tables from their source CSVs (the "dedicated
representation" the guide calls for). They are static reference data of a few KB,
versioned next to the pipeline, so this stays in sync with what `upload_csv.py`
uploads. Everything else in the dashboard comes from MongoDB.

`fact_water_list` is *not* here — it normalises cleanly (`ticker`/`lake`/
`plants`/`region`/`values.mw`) and is read from MongoDB in `queries.py`.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from .data import PROJECT_ROOT, cached, num

DATA_DIR = PROJECT_ROOT / "data" / "industries_data"


def _read_csv(name: str) -> list[dict]:
    """Decode a reference CSV (the exports mix UTF-8 and CP1252) into dict rows."""
    raw = (DATA_DIR / f"{name}.csv").read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        text = raw.decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    # Trailing empty columns ("Location,,,") come through as a None key.
    return [{k: v for k, v in row.items() if k} for row in rows]


def _clean(value):
    """`"29,539"` -> 29539.0; blank -> None."""
    return num(value)


# --------------------------------------------------------------------------
# fact_pdp8_capacity — capacity per power type under three PDP8 scenarios
# --------------------------------------------------------------------------
def pdp8_capacity() -> dict:
    def build():
        rows = _read_csv("fact_pdp8_capacity")
        # Header is "Capacity 2025 (MW)", "Capacity 2030 Th?p (MW)", … — the export
        # dropped the diacritics on "Thấp"; restore them for display.
        scenarios, labels = [], {}
        for column in (rows[0] if rows else {}):
            if column == "Type":
                continue
            key = column.replace(" (MW)", "").strip()
            scenarios.append(column)
            labels[column] = key.replace("Th?p", "Thấp").replace("Capacity ", "")
        data = []
        for row in rows:
            name = (row.get("Type") or "").strip()
            if not name:
                continue
            data.append({
                "type": name,
                "values": {labels[c]: _clean(row.get(c)) for c in scenarios},
            })
        return {
            "scenarios": [labels[c] for c in scenarios],
            "rows": data,
            "unit": "MW",
            "total_type": "Total",
        }

    return cached("fact:pdp8_capacity", build)


# --------------------------------------------------------------------------
# fact_pdp8_capex — investment per phase, split State vs Socialized
# --------------------------------------------------------------------------
def pdp8_capex() -> dict:
    def build():
        rows = []
        for row in _read_csv("fact_pdp8_capex"):
            phase = (row.get("Phase") or "").strip().replace("–", "-")
            if not phase:
                continue
            rows.append({
                "phase": phase,
                "investment_type": (row.get("Investment Type") or "").strip(),
                "total": _clean(row.get("Total Capital (billion USD)")),
                "state": _clean(row.get("State Capital (billion USD)")),
                "socialized": _clean(row.get("Socialized Capital (billion USD)")),
            })
        phases = []
        for row in rows:
            if row["phase"] not in phases:
                phases.append(row["phase"])
        # A phase is only splittable when the State/Socialized columns are filled
        # (2021-2025 carries the total alone).
        split = {
            phase: any(
                r["state"] is not None or r["socialized"] is not None
                for r in rows if r["phase"] == phase
            )
            for phase in phases
        }
        return {"rows": rows, "phases": phases, "split": split, "unit": "USDbn"}

    return cached("fact:pdp8_capex", build)


# --------------------------------------------------------------------------
# fact_price_framework — ceiling price per power source and region
# --------------------------------------------------------------------------
def price_framework() -> dict:
    def build():
        rows = []
        for row in _read_csv("fact_price_framework"):
            source = (row.get("Power Source") or "").strip()
            if not source:
                continue
            rows.append({
                "source": source,
                "region": (row.get("Region") or "").strip(),
                "vnd": _clean(row.get("VND/kWh")),
                "cent": _clean(row.get("US cent/kWh")),
            })
        regions = []
        for row in rows:
            if row["region"] and row["region"] not in regions:
                regions.append(row["region"])
        return {"rows": rows, "regions": regions, "unit": "VND/kWh"}

    return cached("fact:price_framework", build)


# --------------------------------------------------------------------------
# fact_pdp8_plant — the project list behind the plan
# --------------------------------------------------------------------------
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def pdp8_plants() -> dict:
    def build():
        rows = []
        for row in _read_csv("fact_pdp8_plant"):
            project = (row.get("Project") or "").strip()
            if not project:
                continue
            rows.append({
                "project": project,
                "type": (row.get("Type") or "").strip() or "Không rõ",
                "capacity": _clean(row.get("Capacity (MW)")),
                "operation": (row.get("Expected Operation") or "").strip(),
                "progress": (row.get("Progress") or "").strip(),
                "investor": (row.get("Investor") or "").strip(),
                "location": (row.get("Location") or "").strip(),
            })

        by_type: dict[str, dict] = {}
        for row in rows:
            acc = by_type.setdefault(row["type"], {"type": row["type"], "capacity": 0.0, "count": 0})
            acc["capacity"] += row["capacity"] or 0.0
            acc["count"] += 1
        types = sorted(by_type.values(), key=lambda x: -x["capacity"])

        # "Expected Operation" is free text ("2028-2029", "2029") — take the first
        # year mentioned so the table can be grouped by commissioning year too.
        by_year: dict[str, dict] = {}
        for row in rows:
            match = _YEAR_RE.search(row["operation"])
            year = match.group(0) if match else "Chưa rõ"
            acc = by_year.setdefault(year, {"year": year, "capacity": 0.0, "count": 0})
            acc["capacity"] += row["capacity"] or 0.0
            acc["count"] += 1
        years = sorted(by_year.values(), key=lambda x: x["year"])

        return {
            "rows": rows,
            "by_type": types,
            "by_year": years,
            "total_capacity": sum(r["capacity"] or 0.0 for r in rows),
            "unit": "MW",
        }

    return cached("fact:pdp8_plants", build)
