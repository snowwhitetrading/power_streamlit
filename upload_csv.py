import argparse
import hashlib
import numbers
import os
import re
import tomllib
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
from pymongo import MongoClient, UpdateOne


# Both industry-level and company-level CSVs feed the same normalized collection.
# A single run walks every directory listed here so one command keeps MongoDB in
# sync with the whole data/ tree.
DEFAULT_DATA_DIRS = [
    Path("data") / "industries_data",
    Path("data") / "companies_data",
]
DEFAULT_DB_NAME = "dc_commodity"
DEFAULT_TARGET_COLLECTION = "ThiTruongDien"


def load_secret(name: str) -> str:
    """Resolve a secret from the environment, then from .secrets.toml/secrets.toml."""
    env_value = os.getenv(name)
    if env_value:
        return env_value
    here = Path(__file__).resolve().parent
    candidates = [
        Path(".secrets.toml"),
        Path("secrets.toml"),
        here / ".secrets.toml",
        here / "secrets.toml",
    ]
    for path in candidates:
        try:
            if not path.exists():
                continue
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        value = data.get(name)
        if value:
            return str(value)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload CSV data to MongoDB in normalized date/object/value format."
    )
    parser.add_argument(
        "--mongo-uri",
        default=load_secret("MONGO_URI"),
        help="MongoDB connection string. Default: MONGO_URI env var, then .secrets.toml.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        default=DEFAULT_DATA_DIRS,
        help=(
            "One or more root directories that contain CSV files. "
            "Defaults to both data/industries_data and data/companies_data."
        ),
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help="Target MongoDB database name.",
    )
    parser.add_argument(
        "--target-collection",
        default=DEFAULT_TARGET_COLLECTION,
        help="Target collection name, for example ThiTruongDien.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Number of records per bulk write batch.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="If set, delete all existing docs in target collection before upload.",
    )
    parser.add_argument(
        "--keep-legacy",
        action="store_true",
        help="If set, keep old article-style documents that have article_id/content fields.",
    )
    return parser.parse_args()


def iter_csv_files(root: Path) -> Iterable[Path]:
    return sorted(root.rglob("*.csv"))


def read_csv_with_fallback(csv_file: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(csv_file, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error is None:
        raise ValueError(f"Unable to decode CSV file {csv_file}")

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Unable to decode CSV file {csv_file}",
    )


@lru_cache(maxsize=8192)
def clean_column_name(column: str) -> str:
    name = str(column).strip()
    name = re.sub(r"\.\d+$", "", name)
    name = re.sub(r"__dup\d+$", "", name)
    return name


def make_unique_columns(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for raw in columns:
        base = clean_column_name(raw)
        count = seen.get(base, 0)
        if count == 0:
            unique.append(base)
        else:
            unique.append(f"{base}__dup{count + 1}")
        seen[base] = count + 1
    return unique


def normalize_text(value: object) -> str | None:
    if isinstance(value, pd.Series):
        if value.empty:
            return None
        value = value.iloc[0]
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text in {"-", "--"}:
        return None
    return re.sub(r"\s+", " ", text)


def get_first_value(row: dict, *names: str) -> object:
    """Return the first column value that is present and non-empty.

    The fetch_* CSVs now use a unified `date` column; the legacy names
    (thoiGian/report_date/date_time/update_date) are kept as fallbacks.
    """
    for name in names:
        value = row.get(name)
        if normalize_text(value) is not None:
            return value
    return None


def parse_numeric(value: object) -> float | None:
    # Cells read from the CSVs are usually already numeric (numpy int/float);
    # handle those directly instead of round-tripping through normalize_text's
    # regex. bool is excluded so it falls through to the None-returning path.
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        return None if number != number else number  # drop NaN

    text = normalize_text(value)
    if text is None:
        return None
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


# Explicit formats the source CSVs actually use, tried before falling back to
# pandas. pd.to_datetime re-guesses the format on every call and was ~70% of the
# upload's CPU time; strptime on a known format is orders of magnitude cheaper.
_FAST_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
)


@lru_cache(maxsize=8192)
def parse_date_value(value: object) -> datetime | None:
    text = normalize_text(value)
    if text is None:
        return None

    quarter_match = re.fullmatch(r"([1-4])Q(\d{4})", text, flags=re.IGNORECASE)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        month = quarter * 3
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)

    year_match = re.fullmatch(r"\d{4}", text)
    if year_match:
        return datetime(int(text), 12, 31)

    # Fast path for the common ISO / dd-mm-yyyy spellings (the 'T' separator is
    # normalized to a space so both ISO variants hit the same formats).
    candidate = text.replace("T", " ")
    for fmt in _FAST_DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue

    # Fallback for unexpected spellings. Day-first (dd/mm/yyyy) only for slash
    # dates; ISO (yyyy-mm-dd) is parsed as written. Choosing per-text avoids
    # pandas' "dayfirst ignored" warning while keeping both source formats correct.
    parsed = pd.to_datetime(text, errors="coerce", dayfirst="/" in text)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def detect_date_column(columns: list[str]) -> str | None:
    candidates = [
        "date",
        "Date",
        "report_date",
        "query_date",
        "thoiGian",
        "update_date",
        "date_time",
        "Unnamed: 0",
        "Year",
        "year",
    ]
    lowered = {str(c).lower(): str(c) for c in columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


@lru_cache(maxsize=4096)
def split_object_metric_unit(column: str, default_object: str) -> tuple[str, str, str | None]:
    name = clean_column_name(column)
    unit_match = re.search(r"\(([^()]+)\)\s*$", name)
    unit = unit_match.group(1).strip() if unit_match else None
    if unit_match:
        name = name[: unit_match.start()].strip()

    metric_tokens = [
        "gross profit",
        "net revenue",
        "quarterly volume",
        "mobilized",
        "contracted",
        "revenue",
        "asp",
        "volume",
        "npat",
        "gas cost",
        "cost",
        "alpha",
        "temp mean",
        "temp_mean",
        "oni",
    ]

    if " - " in name:
        parts = [p.strip() for p in name.split(" - ", maxsplit=1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1], unit

    lower_name = name.lower()
    for token in metric_tokens:
        idx = lower_name.rfind(token)
        if idx > 0:
            obj = name[:idx].strip(" -_")
            metric = name[idx:].strip()
            if obj and metric:
                return obj, metric, unit

    return default_object, name, unit


def canonical_date_key(date_value: datetime | None, date_text: str | None) -> str:
    """Stable date key for the _id, independent of the raw CSV date *format*.

    The same instant can be spelled several ways ('2020-01-01T00:30:00',
    '2020-01-01 00:30:00', '01/01/2020 06:00', '31/8/2020'). Hashing the raw text
    meant a format change produced a different _id for the same record, leaving the
    old document behind as a duplicate. Normalizing the key to the standardized
    'YYYY-MM-DD[ HH:MM:SS]' form makes re-uploads idempotent across format changes.

    Only values carrying a calendar separator are normalized; special tokens
    ('1Q2024', bare years) and already-ISO strings are returned unchanged so the
    _ids of correctly-stored documents stay stable.
    """
    text = (date_text or "").strip()
    if "/" in text and date_value is not None:
        # dd/mm/yyyy[ HH:MM[:SS]] -> ISO; keep the time component only when present.
        if ":" in text:
            return date_value.strftime("%Y-%m-%d %H:%M:%S")
        return date_value.strftime("%Y-%m-%d")
    # ISO 'T' separators -> space (no-op for the already-standardized form).
    return text.replace("T", " ")


def build_doc_id(
    dataset: str,
    date_key: str | None,
    obj: str,
) -> str:
    payload = f"{dataset}|{date_key or ''}|{obj}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def build_base_doc(
    dataset: str,
    source_file: str,
    date_value: datetime | None,
    date_text: str | None,
    obj: str,
) -> dict:
    return {
        "_id": build_doc_id(
            dataset=dataset,
            date_key=canonical_date_key(date_value, date_text),
            obj=obj,
        ),
        "date": date_value,
        "date_text": date_text,
        "object": obj,
        "dataset": dataset,
        "source_file": source_file,
        "values": {},
        "units": {},
        "updated_at": datetime.now(timezone.utc),
    }


def add_value(doc: dict, key: str, raw_value: object, unit: str | None = None) -> None:
    numeric_value = parse_numeric(raw_value)
    if numeric_value is None:
        return
    metric_key = clean_column_name(key)
    doc["values"][metric_key] = numeric_value
    if unit:
        doc["units"][metric_key] = unit


@lru_cache(maxsize=None)
def generic_column_plan(
    dataset: str,
    columns: tuple[str, ...],
    default_object: str | None = None,
) -> tuple[str | None, tuple[tuple[str, str, str, str | None], ...]]:
    """Precompute the date column and per-column (col, object, metric, unit) split.

    These derive only from the column names, which are identical for every row of a
    file. Computing them once per file (instead of per row) removes the bulk of the
    upload's CPU cost -- the regex-heavy split_object_metric_unit previously ran for
    every column of every row.

    `default_object` is the object assigned to columns whose header carries no
    embedded object (e.g. a bare "Volume (mn kWh)"). It defaults to the humanized
    dataset name, but callers that know a better label -- company files pass the
    ticker -- override it so single-segment series get a meaningful object.
    """
    date_col = detect_date_column(list(columns))
    if default_object is None:
        default_object = dataset.replace("_", " ")
    specs = tuple(
        (col, *split_object_metric_unit(col, default_object))
        for col in columns
        if col != date_col and not str(col).startswith("Unnamed")
    )
    return date_col, specs


def normalize_generic_wide(
    row: dict,
    date_col: str | None,
    col_specs: tuple[tuple[str, str, str, str | None], ...],
    source_file: str,
    dataset: str,
) -> list[dict]:
    grouped: dict[str, dict] = {}
    date_text = normalize_text(row[date_col]) if date_col else None
    date_value = parse_date_value(date_text) if date_text else None

    for col, obj, metric, unit in col_specs:
        if obj not in grouped:
            grouped[obj] = build_base_doc(
                dataset=dataset,
                source_file=source_file,
                date_value=date_value,
                date_text=date_text,
                obj=obj,
            )
        add_value(grouped[obj], metric, row[col], unit)

    return [doc for doc in grouped.values() if doc["values"]]


def normalize_row(
    file_stem: str,
    source_file: str,
    row: dict,
    row_idx: int,
    columns: list[str],
) -> list[dict]:
    lower_stem = file_stem.lower()

    if lower_stem == "fetch_price_smp_monthly":
        dt_text = normalize_text(get_first_value(row, "date", "thoiGian"))
        dt_value = parse_date_value(dt_text)
        if dt_value is None:
            return []

        date_text = dt_value.strftime("%Y-%m-%d")
        time_slot = dt_value.strftime("%H:%M")
        doc = build_base_doc(
            dataset=file_stem,
            source_file=source_file,
            date_value=parse_date_value(date_text),
            date_text=date_text,
            obj="giaBien",
        )

        row_values: dict[str, float] = {}
        for metric in ["giaBienMB", "giaBienMT", "giaBienMN", "giaBienHT"]:
            numeric_value = parse_numeric(row.get(metric))
            if numeric_value is None:
                continue
            row_values[metric] = numeric_value

        if row_values:
            doc["values"][time_slot] = row_values
            doc["units"] = {
                "giaBienMB": "VND/kWh",
                "giaBienMT": "VND/kWh",
                "giaBienMN": "VND/kWh",
                "giaBienHT": "VND/kWh",
            }

        return [doc] if doc["values"] else []

    if lower_stem == "fetch_volume_region_monthly":
        date_text = normalize_text(get_first_value(row, "date", "thoiGian"))
        doc = build_base_doc(
            dataset=file_stem,
            source_file=source_file,
            date_value=parse_date_value(date_text),
            date_text=date_text,
            obj="congSuat",
        )
        add_value(doc, "congSuatMB", row.get("congSuatMB"), "MW")
        add_value(doc, "congSuatMT", row.get("congSuatMT"), "MW")
        add_value(doc, "congSuatMN", row.get("congSuatMN"), "MW")
        add_value(doc, "congSuatHT", row.get("congSuatHT"), "MW")
        return [doc] if doc["values"] else []

    if lower_stem == "fetch_capacity_mobilized_monthly":
        date_text = normalize_text(get_first_value(row, "date", "report_date"))
        obj = normalize_text(row.get("source")) or "unknown"
        doc = build_base_doc(
            dataset=file_stem,
            source_file=source_file,
            date_value=parse_date_value(date_text),
            date_text=date_text,
            obj=obj,
        )
        add_value(doc, "mobilized_low_season", row.get("low_season"), "MW")
        add_value(doc, "mobilized_high_season", row.get("high_season"), "MW")
        return [doc] if doc["values"] else []

    if lower_stem == "fetch_capacity_installed_monthly":
        date_text = normalize_text(get_first_value(row, "date", "report_date"))
        obj = normalize_text(row.get("source")) or "unknown"
        doc = build_base_doc(
            dataset=file_stem,
            source_file=source_file,
            date_value=parse_date_value(date_text),
            date_text=date_text,
            obj=obj,
        )
        add_value(doc, "installed_capacity", row.get("installed_total_mw"), "MW")
        add_value(doc, "low_peak", row.get("low_peak_mw"), "MW")
        add_value(doc, "high_peak", row.get("high_peak_mw"), "MW")
        return [doc] if doc["values"] else []

    if lower_stem == "fetch_hydro_reservoir_monthly":
        date_text = normalize_text(get_first_value(row, "date", "date_time"))
        obj = normalize_text(row.get("reservoir_name")) or "unknown"
        doc = build_base_doc(
            dataset=file_stem,
            source_file=source_file,
            date_value=parse_date_value(date_text),
            date_text=date_text,
            obj=obj,
        )
        add_value(doc, "flood_level", row.get("flood_level"), "m")
        add_value(doc, "flood_capacity", row.get("flood_capacity"), "%")
        add_value(doc, "plant_throughput", row.get("plant_throughput"), "m3/s")
        return [doc] if doc["values"] else []

    if lower_stem == "fetch_coal_vinacomin_monthly":
        date_text = normalize_text(get_first_value(row, "date", "update_date"))
        obj = normalize_text(row.get("commodity")) or "unknown"
        price_unit = normalize_text(row.get("currency_per_unit"))
        doc = build_base_doc(
            dataset=file_stem,
            source_file=source_file,
            date_value=parse_date_value(date_text),
            date_text=date_text,
            obj=obj,
        )
        add_value(doc, "price", row.get("price"), price_unit)
        add_value(doc, "percent_change", row.get("percent_change"), "%")
        return [doc] if doc["values"] else []

    if lower_stem.startswith("company_"):
        # Per-company wide time-series: company_{ticker}_{freq}.csv. Same
        # "Object Metric (Unit)" column layout the generic path handles, but every
        # resulting doc is tagged with the ticker parsed from the filename so the
        # collection stays queryable by company without parsing the dataset name.
        # The ticker is also the default object so single-segment files (bare
        # "Volume"/"Revenue" columns, e.g. HND/PPC/QTP) group under "HND" instead
        # of the humanized dataset name; multi-segment files (GEG's "Solar Volume")
        # still split their object out of the column header.
        parts = file_stem.split("_")
        ticker = parts[1].upper() if len(parts) > 1 and parts[1] else None
        date_col, col_specs = generic_column_plan(
            file_stem, tuple(columns), ticker or file_stem.replace("_", " ")
        )
        # Drop any cadence word the split left on the object (PGV plants come out as
        # "Phu My Monthly" -> "Phu My"); the original column name is kept for the
        # row lookup, only the object label is cleaned so the _id derives from it.
        col_specs = tuple(
            (col, strip_cadence_word(obj), metric, unit)
            for col, obj, metric, unit in col_specs
        )
        docs = normalize_generic_wide(
            row=row,
            date_col=date_col,
            col_specs=col_specs,
            source_file=source_file,
            dataset=file_stem,
        )
        for doc in docs:
            doc["ticker"] = ticker
        return docs

    if lower_stem == "fact_water_list":
        # Reference/dimension table (ticker, region, lake, plants, mw) with no
        # date column. Kept as one document per reservoir row: the text columns
        # become top-level attributes and the capacity is the only numeric metric.
        # The object embeds the ticker because reservoirs recur across tickers
        # (e.g. "Sông Hinh", "Thác Bà", "Buôn Kuốp"); a lake-only object would let
        # the (dataset, date, object) dedupe pass collapse those distinct rows.
        ticker = normalize_text(row.get("ticker"))
        lake = normalize_text(row.get("lake"))
        if not ticker and not lake:
            return []
        obj = " - ".join(part for part in (ticker, lake) if part) or "unknown"
        doc = build_base_doc(
            dataset=file_stem,
            source_file=source_file,
            date_value=None,
            date_text=None,
            obj=obj,
        )
        doc["ticker"] = ticker
        doc["region"] = normalize_text(row.get("region"))
        doc["lake"] = lake
        doc["plants"] = normalize_text(row.get("plants"))
        add_value(doc, "mw", row.get("mw"), "MW")
        return [doc]

    if lower_stem == "fact_pdp8_capacity":
        records: list[dict] = []
        obj = normalize_text(row.get("Type")) or "unknown"
        for col in columns:
            if col == "Type":
                continue
            year_match = re.search(r"(\d{4})", col)
            year = year_match.group(1) if year_match else None
            if not year:
                continue
            date_text = f"{year}-12-31"
            doc = build_base_doc(
                dataset=file_stem,
                source_file=source_file,
                date_value=parse_date_value(date_text),
                date_text=date_text,
                obj=obj,
            )
            _, metric, unit = split_object_metric_unit(col, "capacity")
            add_value(doc, metric, row.get(col), unit)
            if doc["values"]:
                records.append(doc)
        return records

    date_col, col_specs = generic_column_plan(file_stem, tuple(columns))
    return normalize_generic_wide(
        row=row,
        date_col=date_col,
        col_specs=col_specs,
        source_file=source_file,
        dataset=file_stem,
    )


def upload_csv_to_collection(
    csv_file: Path,
    data_root: Path,
    db,
    target_collection: str,
    batch_size: int,
) -> tuple[int, int]:
    df = read_csv_with_fallback(csv_file)
    source_file = str(csv_file.relative_to(data_root)).replace("\\", "/")
    collection = db[target_collection]
    # Replace-in-place per dataset. Every document a file produces carries
    # dataset == the file stem, and the CSV holds the complete history for that
    # series, so clearing the dataset first makes this upload authoritative. If the
    # object/date derivation changed since a prior run (renamed columns, a new date
    # spelling), the superseded documents would otherwise linger as silent overlap
    # under a different object/_id that neither the upsert nor the dedupe catches.
    collection.delete_many({"dataset": csv_file.stem})
    columns = make_unique_columns([str(c) for c in df.columns])
    df.columns = columns
    # to_dict("records") iterates plain dicts, avoiding the per-row Series that
    # df.iterrows() builds -- markedly faster for the row-by-row normalization.
    records = df.to_dict("records")

    operations = []
    rows_read = 0
    metric_docs = 0

    if csv_file.stem.lower() == "fetch_price_smp_monthly":
        grouped_docs: dict[str, dict] = {}
        for row_idx, row in enumerate(records):
            docs = normalize_row(
                file_stem=csv_file.stem,
                source_file=source_file,
                row=row,
                row_idx=row_idx,
                columns=columns,
            )
            rows_read += 1
            if not docs:
                continue

            doc = docs[0]
            existing = grouped_docs.get(doc["_id"])
            if existing is None:
                grouped_docs[doc["_id"]] = doc
                continue

            for slot_key, slot_values in doc["values"].items():
                existing.setdefault("values", {})[slot_key] = slot_values
            for metric_key, unit in doc.get("units", {}).items():
                existing.setdefault("units", {})[metric_key] = unit
            existing["updated_at"] = datetime.now(timezone.utc)

        for doc in grouped_docs.values():
            operations.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
            metric_docs += 1

            if len(operations) >= batch_size:
                collection.bulk_write(operations, ordered=False)
                operations.clear()

        if operations:
            collection.bulk_write(operations, ordered=False)

        return rows_read, metric_docs

    for row_idx, row in enumerate(records):
        docs = normalize_row(
            file_stem=csv_file.stem,
            source_file=source_file,
            row=row,
            row_idx=row_idx,
            columns=columns,
        )
        rows_read += 1

        for doc in docs:
            operations.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
            metric_docs += 1

        if len(operations) >= batch_size:
            collection.bulk_write(operations, ordered=False)
            operations.clear()

    if operations:
        collection.bulk_write(operations, ordered=False)

    return rows_read, metric_docs


# Frequency suffixes a source file name may carry. The part before the suffix is
# the *logical source*; only the suffix distinguishes, say, a monthly dump from a
# later quarterly re-cut of the SAME series.
_FREQ_SUFFIX_RE = re.compile(
    r"_(?:monthly|quarterly|annually|yearly|annual|daily|weekly)$",
    flags=re.IGNORECASE,
)


# A cadence word embedded in a column header (e.g. PGV's "Phu My Monthly Volume")
# leaks into the derived object as a trailing "... Monthly". Strip it so the object
# is the clean entity name ("Phu My"), matching how every other company file reads.
_CADENCE_WORD_RE = re.compile(
    r"\s+(?:monthly|quarterly|annually|yearly|annual|daily|weekly)\b",
    flags=re.IGNORECASE,
)


def strip_cadence_word(name: str) -> str:
    return _CADENCE_WORD_RE.sub("", name).strip()


def dataset_family(dataset: str) -> str:
    """Identity of a series that survives a frequency-suffix rename.

    'company_geg_monthly' and 'company_geg_quarterly' are the same underlying
    company series recut at a different cadence, so both map to 'company_geg'.
    A dataset with no frequency suffix (e.g. 'fact_water_list') is its own family.
    """
    return _FREQ_SUFFIX_RE.sub("", dataset)


def prune_stale_dataset_twins(collection, uploaded_datasets: Iterable[str]) -> int:
    """Delete previously-stored datasets that this run replaced under a new name.

    The `dataset` value (the CSV file stem) is baked into every `_id`, so renaming
    a file -- e.g. company_geg_monthly.csv -> company_geg_quarterly.csv -- makes the
    new upload land under a *new* `_id` while the old documents linger untouched.
    The result is the same company present twice, under two dataset names: exactly
    the overlap this guards against.

    For every dataset we just wrote, any *other* stored dataset sharing its
    `dataset_family` (same logical source, different frequency suffix) is a stale
    twin and gets removed. The prune is scoped to families we actually uploaded, so
    running on a subset of the data (e.g. only companies_data) never touches
    unrelated series, and genuinely removed sources with no current counterpart are
    left alone rather than silently wiped.
    """
    uploaded = set(uploaded_datasets)
    uploaded_families = {dataset_family(ds) for ds in uploaded}

    stale = [
        ds
        for ds in collection.distinct("dataset")
        if ds not in uploaded and dataset_family(ds) in uploaded_families
    ]
    if not stale:
        return 0
    return collection.delete_many({"dataset": {"$in": stale}}).deleted_count


def dedupe_logical_duplicates(collection) -> int:
    """Collapse logical duplicates left behind by historical _id drift.

    The _id is a hash of (dataset, canonical date_key, object). Whenever that
    derivation changed across code versions or CSV date formats, the same
    logical record acquired a *new* _id while the previous document lingered,
    because the upsert matches only on _id. With no unique index on the logical
    key, those orphans accumulated as duplicates on every such change.

    This pass keeps the most recently written document per
    (dataset, date, object) and deletes the older twins. Records with no
    duplicate (count == 1) are never touched, so genuinely distinct historical
    rows -- e.g. the rolling date coverage of the fetch_* datasets -- are
    preserved. Running it after every upload makes re-uploads idempotent
    regardless of any future _id-derivation change.
    """
    pipeline = [
        {"$sort": {"updated_at": -1}},
        {
            "$group": {
                "_id": {"dataset": "$dataset", "date": "$date", "object": "$object"},
                "ids": {"$push": "$_id"},
                "count": {"$sum": 1},
            }
        },
        {"$match": {"count": {"$gt": 1}}},
        {"$project": {"losers": {"$slice": ["$ids", 1, 1_000_000]}}},
    ]

    loser_ids: list = []
    for group in collection.aggregate(pipeline, allowDiskUse=True):
        loser_ids.extend(group["losers"])

    deleted = 0
    for start in range(0, len(loser_ids), 10_000):
        chunk = loser_ids[start : start + 10_000]
        deleted += collection.delete_many({"_id": {"$in": chunk}}).deleted_count
    return deleted


def main() -> None:
    args = parse_args()

    if not args.mongo_uri:
        raise ValueError(
            "Missing MongoDB URI. Use --mongo-uri or set MONGO_URI environment variable."
        )

    data_dirs = args.data_dir if isinstance(args.data_dir, list) else [args.data_dir]

    missing = [str(d) for d in data_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(f"Data directory not found: {', '.join(missing)}")

    # Collect (csv_file, source_root) pairs across every requested directory.
    # source_root is the directory's parent so source_file keeps the subdir
    # prefix (e.g. industries_data/... , companies_data/...), which stays unique
    # even when the two trees hold same-named files.
    jobs = []
    for data_dir in data_dirs:
        files = list(iter_csv_files(data_dir))
        if not files:
            print(f"No CSV files found in {data_dir}")
            continue
        for csv_file in files:
            jobs.append((csv_file, data_dir.parent))

    if not jobs:
        print(f"No CSV files found in {', '.join(str(d) for d in data_dirs)}")
        return

    client = MongoClient(args.mongo_uri)
    db = client[args.db_name]
    collection = db[args.target_collection]

    if args.replace:
        collection.delete_many({})

    if not args.keep_legacy:
        deleted = collection.delete_many({"article_id": {"$exists": True}}).deleted_count
        if deleted:
            print(f"Deleted {deleted} legacy article-style docs from {args.target_collection}")

    total_rows = 0
    total_docs = 0

    for csv_file, source_root in jobs:
        rows, docs = upload_csv_to_collection(
            csv_file=csv_file,
            data_root=source_root,
            db=db,
            target_collection=args.target_collection,
            batch_size=args.batch_size,
        )
        total_rows += rows
        total_docs += docs
        print(f"Processed {rows} source rows -> {docs} metric docs from {csv_file}")

    collection.create_index([("date", 1), ("object", 1), ("metric", 1)])
    collection.create_index([("source_file", 1), ("dataset", 1)])

    uploaded_datasets = {csv_file.stem for csv_file, _ in jobs}
    pruned = prune_stale_dataset_twins(collection, uploaded_datasets)
    if pruned:
        print(
            f"Removed {pruned} documents from stale dataset twins "
            "(same source re-uploaded under a renamed frequency suffix)."
        )

    removed = dedupe_logical_duplicates(collection)
    if removed:
        print(f"Removed {removed} duplicate documents left by earlier _id formats.")

    print(
        f"Completed upload: {total_rows} source rows converted into {total_docs} metric docs "
        f"in '{args.db_name}.{args.target_collection}'."
    )


if __name__ == "__main__":
    main()
