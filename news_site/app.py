"""News/Updates website for the power-sector news stored in MongoDB.

Serves a single-page feed (index.html) styled like an "updates" changelog and a
JSON API (/api/news) that reads the `TinNganhDien` collection of the
`dc_commodity` database — the same collection `upload_json.py` populates.

Run (from the project root):
    .venv/Scripts/python.exe -m uvicorn news_site.app:app --reload --port 8000
then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import re
import tomllib
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from pymongo import MongoClient

DB_NAME = "dc_commodity"
COLLECTION = "TinNganhDien"
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


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
def get_collection():
    uri = load_secret("MONGO_URI")
    if not uri:
        raise RuntimeError("Missing MONGO_URI (set env var or add it to .secrets.toml).")
    return MongoClient(uri)[DB_NAME][COLLECTION]


app = FastAPI(title="Tin ngành điện")


# --- Tag normalization -----------------------------------------------------
# The feeds spell the same entity several ways (ticker code, company long-name,
# search keyword). Collapse each group into ONE canonical tag for the filter.
# Matching is case-insensitive (casefold); keep the exact diacritics here.
TICKER_ALIASES = {
    "TV2": ["TV2", "PECC2", "Tư vấn Xây dựng Điện 2"],
    "POW": ["POW", "PV POW"],
    "PLX": ["PLX", "Petrolimex"],
    "OIL": ["OIL", "PV OIL"],
    "GAS": ["GAS", "PV GAS"],
    "HDG": ["HDG", "Tập đoàn Hà Đô"],
    "PVD": ["PVD", "PV Drilling"],
    "PVS": ["PVS", "PTSC"],
    "GEG": ["GEG", "Điện Gia Lai GEG"],
    "NT2": ["NT2", "Điện lực Dầu khí Nhơn Trạch 2"],
    "PGV": ["PGV", "EVNGenco3 PGV", "EVNGENCO3"],
}
KEYWORD_ALIASES = {
    "Nhơn Trạch 3 và 4": ["Nhơn Trạch 3", "Nhơn Trạch 4"],
}
DROP_TAGS = {"tư vấn điện"}  # removed from the filter entirely

_REVERSE: dict[str, tuple[str, str]] = {}  # variant.casefold() -> (canonical, kind)
_EXPAND: dict[str, list[str]] = {}         # canonical -> [raw variants...]
for _canon, _variants in {**TICKER_ALIASES, **KEYWORD_ALIASES}.items():
    _EXPAND[_canon] = _variants
    _kind = "ticker" if _canon in TICKER_ALIASES else "keyword"
    for _v in _variants:
        _REVERSE[_v.casefold()] = (_canon, _kind)
_DROP = {d.casefold() for d in DROP_TAGS}


def canonical_tag(raw):
    """Map a raw tag to (canonical, kind|None). Returns (None, None) if dropped."""
    if not raw:
        return None, None
    key = raw.strip().casefold()
    if key in _DROP:
        return None, None
    if key in _REVERSE:
        return _REVERSE[key]
    return raw, None  # unknown tag kept as-is; kind decided by where it occurs


def expand_tag(canon: str) -> list[str]:
    """All raw variants a canonical tag should match in a query."""
    return _EXPAND.get(canon, [canon])


def canonical_tags(raw_list) -> list[str]:
    """Canonicalize + dedupe a list of raw tags for display (drops removed ones)."""
    out: list[str] = []
    for raw in raw_list or []:
        canon, _ = canonical_tag(raw)
        if canon and canon not in out:
            out.append(canon)
    return out


# Energy/power-sector vocabulary. A news article counts as "relevant" if it either
# matched a financial topic (score/topics) OR mentions the sector — this keeps
# industry & policy news (E10 rollout, ministry statements, an industry legal case)
# that a company-centric score misses, while off-topic substring matches (a shoe
# tagged "REE", a Ford Ranger tagged "OIL") drop out. Terms are multi-word or
# unambiguous to avoid matching "điện thoại"/"dầu máy".
SECTOR_KEYWORDS = [
    # Năng lượng / dầu khí (multi-word to avoid matching "điện thoại"/"dầu máy")
    "xăng", "dầu khí", "dầu thô", "xăng dầu", "xăng sinh học", "lọc dầu",
    "khí đốt", "khí lng", "năng lượng", "than đá",
    "thủy điện", "nhiệt điện", "điện gió", "điện mặt trời", "điện khí",
    "phát điện", "lưới điện", "ngành điện", "giá điện", "quy hoạch điện",
    "tiền điện", "cắt điện", "sản lượng điện", "thiếu điện", "tiêu thụ điện",
    "điện lực", "nhà máy điện", "dự án điện", "công suất điện",
    "evn", "pvn", "petrolimex", "e10",
    # Sự kiện doanh nghiệp / tài chính (nới rộng để tin doanh nghiệp không bị miss)
    "cổ tức", "cổ phiếu", "trái phiếu", "niêm yết", "chào bán", "cổ đông",
    "đhđcđ", "đại hội đồng cổ đông", "doanh thu", "lợi nhuận", "lãi ròng",
    "kết quả kinh doanh", "báo cáo tài chính", "vốn điều lệ", "thoái vốn",
    # Dự án / hợp đồng / pháp lý
    "dự án", "hợp đồng", "epc", "nhà thầu", "khởi công", "đầu tư", "đấu thầu",
    "nhà máy", "khởi tố", "thanh tra", "chủ trương đầu tư",
]
SECTOR_RE = "|".join(SECTOR_KEYWORDS)

# Advertorial / PR sources that merely name-drop a company (e.g. a vendor bragging
# it upgraded a plant's meeting-room audio) — not real news. They score like real
# company news (company mention + a sector word) so nothing else filters them; the
# only reliable signal is the publisher. Add new ones here as they surface.
BLOCK_SOURCES = [
    # Advertorial / vendor PR (name-drops a company, not real news)
    "Trường Thịnh Company",
    # Social-media platforms — never a financial-news publisher; a keyword match
    # here (a TikTok account named "…Giá Xăng Dầu", a YouTube local-TV roundup) is
    # noise regardless of score/topic.
    "Facebook", "YouTube", "TikTok", "Instagram", "Threads", "Zalo",
    # Job listings, AI aggregators, entertainment, foreign-language false matches.
    "vieclam.tuoitre.vn", "AI Hay", "Yeah1 News", "Latvijas Valsts",
]
# Matched case-insensitively as a substring of the source name.
BLOCK_SOURCE_RE = "|".join(re.escape(s) for s in BLOCK_SOURCES)

# Off-topic content that sneaks in via substring keyword matches — e.g. a film
# project "dự án điện ẢNH" matches the "dự án điện" keyword. When the title/snippet
# reads as entertainment, drop it from the default view (still shown under "Tất cả").
NEG_KEYWORDS = [
    "điện ảnh", "phim", "đạo diễn", "diễn viên", "ca sĩ", "showbiz",
    "hoa hậu", "người mẫu", "Hollywood", "teaser", "phòng vé", "bom tấn",
]
NEG_RE = "|".join(re.escape(k) for k in NEG_KEYWORDS)


def serialize(doc: dict) -> dict:
    """Turn a Mongo document into a JSON-friendly dict for the frontend."""
    date = doc.get("date")
    return {
        "id": doc.get("_id"),
        "type": doc.get("type"),
        "title": doc.get("title"),
        "snippet": doc.get("snippet"),
        "url": doc.get("url"),
        "source": doc.get("source"),
        "date": date.strftime("%Y-%m-%d") if date else (doc.get("date_text") or ""),
        # Company tags come from `tickers` (document/press) or `search` (news),
        # collapsed to their canonical form for display.
        "tags": canonical_tags(doc.get("tickers") or doc.get("search") or []),
        "importance": doc.get("importance_score"),
        "is_important": bool(doc.get("is_important")),
        "topics": doc.get("matched_topics") or [],
    }


@app.get("/api/news")
def api_news(
    type: str = Query("all", description="all | industry | company | document | press"),
    q: str = Query("", description="free-text search over title/snippet"),
    tag: str = Query("", description="filter by a ticker/keyword tag"),
    date_from: str = Query("", description="inclusive lower bound, YYYY-MM-DD"),
    date_to: str = Query("", description="inclusive upper bound, YYYY-MM-DD"),
    relevance: str = Query("relevant", description="relevant | important | all"),
    skip: int = Query(0, ge=0),
    limit: int = Query(30, ge=1, le=100),
):
    """Paginated, filtered news feed — newest first.

    The four sections split the feed by kind:
      industry = news articles with NO company ticker (sector-wide themes),
      company  = news articles tagged to a ticker,
      document = IR disclosures,
      press    = company press releases.
    """
    col = get_collection()
    query: dict = {}
    has_ticker = {"tickers.0": {"$exists": True}}
    no_ticker = {"tickers.0": {"$exists": False}}
    if type == "document":
        query["type"] = "document"
    elif type == "press":
        query["type"] = "press"
    elif type == "industry":
        query.update({"type": "news"}, **no_ticker)
    elif type == "company":
        query.update({"type": "news"}, **has_ticker)
    # type == "all" -> no type restriction

    # Relevance gate. Only the `news` feed is scored (importance_score / is_important);
    # documents and press are always kept since they are inherent company disclosures.
    # A short ticker like "REE" can match a substring ("jim-nah-ree-nah") and produce
    # an off-topic article — those score low (no matched topic) and drop out here.
    keep_non_news = {"type": {"$in": ["document", "press"]}}
    if relevance == "important":
        query.setdefault("$and", []).append(
            {"$or": [{"is_important": True}, keep_non_news]}
        )
    elif relevance == "relevant":
        sector_match = {"$or": [
            {"title": {"$regex": SECTOR_RE, "$options": "i"}},
            {"snippet": {"$regex": SECTOR_RE, "$options": "i"}},
        ]}
        relevant_and = [
            {"type": "news"},
            {"is_noise": {"$ne": True}},
            {"$or": [
                {"importance_score": {"$gte": 3}},   # cleared a financial topic
                {"matched_topics.0": {"$exists": True}},
                sector_match,                        # or is genuine sector/policy news
            ]},
        ]
        if BLOCK_SOURCE_RE:
            # Exclude advertorial/PR publishers (a null/absent source still passes).
            relevant_and.append({"source": {"$not": {"$regex": BLOCK_SOURCE_RE, "$options": "i"}}})
        if NEG_RE:
            # Exclude entertainment/off-topic content that matched a keyword by
            # substring (e.g. a film "dự án điện ảnh" matching "dự án điện").
            relevant_and.append({"$nor": [
                {"title": {"$regex": NEG_RE, "$options": "i"}},
                {"snippet": {"$regex": NEG_RE, "$options": "i"}},
            ]})
        relevant_news = {"$and": relevant_and}
        query.setdefault("$and", []).append({"$or": [keep_non_news, relevant_news]})
    # relevance == "all" -> no gate (full high-recall set)
    if tag:
        # Expand the canonical tag to all its raw spellings, then match either a
        # ticker (document/press/company-news) or a search keyword (news).
        aliases = expand_tag(tag)
        query.setdefault("$and", []).append(
            {"$or": [{"tickers": {"$in": aliases}}, {"search": {"$in": aliases}}]}
        )
    if q:
        rx = {"$regex": q, "$options": "i"}
        query.setdefault("$and", []).append(
            {"$or": [{"title": rx}, {"snippet": rx}]}
        )

    # Date range on the parsed `date` field. `date_to` is inclusive: match the whole
    # day by taking everything before the following midnight.
    date_q: dict = {}
    try:
        if date_from:
            date_q["$gte"] = datetime.strptime(date_from, "%Y-%m-%d")
        if date_to:
            date_q["$lt"] = datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)
    except ValueError:
        date_q = {}  # ignore malformed dates rather than 500
    if date_q:
        query["date"] = date_q

    cursor = (
        col.find(query)
        # date keeps the day grouping; first_seen floats freshly-scraped news to the
        # top within a day (missing on older docs -> they sort below); score breaks ties.
        .sort([("date", -1), ("first_seen", -1), ("importance_score", -1)])
        .skip(skip)
        .limit(limit)
    )
    items = [serialize(d) for d in cursor]
    total = col.count_documents(query)
    return JSONResponse({"items": items, "total": total, "skip": skip, "limit": limit})


@app.get("/api/tags")
def api_tags():
    """Filter options, split into two ordered groups: tickers first, then keywords.

    A tag is a *ticker* if it ever appears in a `tickers[]` array (company codes on
    document/press/company-news); everything left in `search[]` is a *keyword*.
    `count` = number of documents a selection would surface (tickers OR search match).
    """
    col = get_collection()
    counts: dict[str, int] = {}       # canonical -> document count
    forced_kind: dict[str, str] = {}  # canonical -> kind fixed by the alias map
    as_ticker: set[str] = set()       # canonicals that ever appear in tickers[]
    for d in col.find({}, {"tickers": 1, "search": 1}):
        in_doc: set[str] = set()
        for t in d.get("tickers") or []:
            canon, kind = canonical_tag(t)
            if not canon:
                continue
            in_doc.add(canon)
            as_ticker.add(canon)
            if kind:
                forced_kind[canon] = kind
        for s in d.get("search") or []:
            canon, kind = canonical_tag(s)
            if not canon:
                continue
            in_doc.add(canon)
            if kind:
                forced_kind[canon] = kind
        for c in in_doc:
            counts[c] = counts.get(c, 0) + 1

    def kind_of(c: str) -> str:
        # Alias map wins; otherwise a tag is a ticker if it ever occurs in tickers[].
        return forced_kind.get(c) or ("ticker" if c in as_ticker else "keyword")

    tickers = sorted(
        ({"tag": c, "count": n} for c, n in counts.items() if kind_of(c) == "ticker"),
        key=lambda x: (-x["count"], x["tag"]),
    )
    keywords = sorted(
        ({"tag": c, "count": n} for c, n in counts.items() if kind_of(c) == "keyword"),
        key=lambda x: (-x["count"], x["tag"]),
    )
    return JSONResponse({"tickers": tickers, "keywords": keywords})


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")
