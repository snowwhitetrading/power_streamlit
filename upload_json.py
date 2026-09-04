"""Upload the company news/document JSON feeds to MongoDB.

The three ``fetch_*_companies.json`` files produced by the processors carry
article-style records (title/snippet/url/date), not the numeric time-series that
``upload_csv.py`` pushes into ``ThiTruongDien``. This script normalizes all
three into one common article shape and upserts them into a dedicated
collection, so re-running the pipeline is idempotent.

Sources handled:
    fetch_documents_companies.json  -> type "document"  (IR disclosures)
    fetch_news_companies.json       -> type "news"      (press coverage)
    fetch_press_companies.json      -> type "press"     (company press releases)

Usage (from the project root):
    python upload_json.py
    python upload_json.py --replace          # wipe the collection first
    python upload_json.py --mongo-uri "mongodb+srv://..."
"""

import argparse
import hashlib
import os
import re
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient, UpdateOne


DEFAULT_DATA_DIR = Path("data")
DEFAULT_DB_NAME = "dc_commodity"
DEFAULT_TARGET_COLLECTION = "TinNganhDien"

# The "freshly-scraped news floats to the top" ordering (first_seen) starts here.
# Runs before this date do NOT stamp first_seen, so 2026-07-08 and earlier stay
# ordered uniformly (by score) and only 2026-07-09+ gets the new-on-top behaviour.
FIRST_SEEN_FROM = datetime(2026, 7, 9, tzinfo=timezone.utc)

# Each source file: its filename, the key holding the item list, and the "type"
# tag stamped on every resulting document so the frontend can filter by kind.
DATASETS = [
    {"file": "fetch_documents_companies.json", "items_key": "documents", "type": "document"},
    {"file": "fetch_news_companies.json", "items_key": "news", "type": "news"},
    {"file": "fetch_press_companies.json", "items_key": "news", "type": "press"},
]


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
        description="Upload company news/document JSON feeds to MongoDB."
    )
    parser.add_argument(
        "--mongo-uri",
        default=load_secret("MONGO_URI"),
        help="MongoDB connection string. Default: MONGO_URI env var, then .secrets.toml.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory that holds the fetch_*_companies.json files (default: data/).",
    )
    parser.add_argument(
        "--db-name",
        default=DEFAULT_DB_NAME,
        help="Target MongoDB database name.",
    )
    parser.add_argument(
        "--target-collection",
        default=DEFAULT_TARGET_COLLECTION,
        help="Target collection for the article documents.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records per bulk write batch.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="If set, delete every existing doc in the target collection before upload.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"-", "--"}:
        return None
    return re.sub(r"\s+", " ", text)


# The feeds spell dates two ways: dd/mm/yyyy (documents/press) and yyyy-mm-dd
# (news). Try both explicitly; return None rather than guessing on anything else.
_DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S")


def parse_date_value(value: object) -> datetime | None:
    text = normalize_text(value)
    if text is None:
        return None
    candidate = text.split("+")[0].strip()  # drop any timezone suffix
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [t for t in (normalize_text(v) for v in value) if t]
    text = normalize_text(value)
    return [text] if text else []


def build_doc_id(dataset: str, url: str, title: str) -> str:
    # URL is the natural unique key; fall back to the title so records with an
    # empty url still get a stable, collision-resistant id.
    key = url or title or ""
    payload = f"{dataset}|{key}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def normalize_item(raw: dict, dataset: str, doc_type: str, source_file: str) -> dict | None:
    title = normalize_text(raw.get("title"))
    url = normalize_text(raw.get("url")) or normalize_text(raw.get("link"))
    if not title and not url:
        return None  # nothing worth storing

    date_text = raw.get("date") or raw.get("news_date")
    query_date_text = raw.get("query_date")

    doc = {
        "_id": build_doc_id(dataset, url or "", title or ""),
        "dataset": dataset,
        "type": doc_type,
        "title": title,
        "snippet": normalize_text(raw.get("snippet")),
        "url": url,
        "source": normalize_text(raw.get("source")),
        "tickers": as_list(raw.get("ticker")),
        "date": parse_date_value(date_text),
        "date_text": normalize_text(date_text),
        "source_file": source_file,
        "updated_at": datetime.now(timezone.utc),
    }

    # News-only fields; kept off the document entirely when absent.
    search = as_list(raw.get("search"))
    if search:
        doc["search"] = search
    if raw.get("query_date") is not None:
        doc["query_date"] = parse_date_value(query_date_text)
        doc["query_date_text"] = normalize_text(query_date_text)
    if raw.get("importance_score") is not None:
        doc["importance_score"] = raw.get("importance_score")
    # Ranking/filtering signals from the news classifier (absent on document/press
    # feeds). is_important marks analyst-grade events; is_noise marks daily
    # price-tracker clutter; matched_topics explains the score for the frontend.
    if raw.get("is_important") is not None:
        doc["is_important"] = bool(raw.get("is_important"))
    if raw.get("is_noise") is not None:
        doc["is_noise"] = bool(raw.get("is_noise"))
    if raw.get("matched_topics"):
        doc["matched_topics"] = as_list(raw.get("matched_topics"))

    return doc


def upload_dataset(spec: dict, data_dir: Path, collection, batch_size: int) -> tuple[int, int]:
    path = data_dir / spec["file"]
    if not path.exists():
        print(f"Skipping {spec['file']}: file not found in {data_dir}")
        return 0, 0

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get(spec["items_key"], [])
    dataset = path.stem
    # Stamp when a doc is FIRST inserted (never overwritten on later runs), so the
    # frontend can float freshly-discovered news above previously-scraped items.
    # Only active from FIRST_SEEN_FROM onward (see constant).
    run_now = datetime.now(timezone.utc)
    stamp_first_seen = run_now >= FIRST_SEEN_FROM

    operations: list[UpdateOne] = []
    read = 0
    written = 0
    seen_ids: set[str] = set()

    for raw in items:
        read += 1
        doc = normalize_item(raw, dataset, spec["type"], spec["file"])
        if doc is None:
            continue
        # De-dupe within a single file so two rows sharing a url/title don't
        # collide inside one unordered bulk write.
        if doc["_id"] in seen_ids:
            continue
        seen_ids.add(doc["_id"])

        update: dict = {"$set": doc}
        if stamp_first_seen:
            update["$setOnInsert"] = {"first_seen": run_now}
        operations.append(UpdateOne({"_id": doc["_id"]}, update, upsert=True))
        written += 1
        if len(operations) >= batch_size:
            collection.bulk_write(operations, ordered=False)
            operations.clear()

    if operations:
        collection.bulk_write(operations, ordered=False)

    print(f"Processed {read} items -> {written} docs from {spec['file']} (dataset '{dataset}')")
    return read, written


def main() -> None:
    args = parse_args()

    if not args.mongo_uri:
        raise ValueError(
            "Missing MongoDB URI. Use --mongo-uri or set MONGO_URI environment variable."
        )
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    client = MongoClient(args.mongo_uri)
    collection = client[args.db_name][args.target_collection]

    if args.replace:
        removed = collection.delete_many({}).deleted_count
        print(f"Replaced: cleared {removed} existing docs from {args.target_collection}")

    total_read = 0
    total_written = 0
    for spec in DATASETS:
        read, written = upload_dataset(spec, args.data_dir, collection, args.batch_size)
        total_read += read
        total_written += written

    collection.create_index([("dataset", 1), ("date", -1)])
    collection.create_index([("type", 1), ("date", -1)])
    collection.create_index([("tickers", 1), ("date", -1)])

    print(
        f"Completed upload: {total_read} source items -> {total_written} docs "
        f"in '{args.db_name}.{args.target_collection}'."
    )


if __name__ == "__main__":
    main()
