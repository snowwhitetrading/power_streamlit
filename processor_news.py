import argparse
import json
import os
import re
import time
import tomllib
import unicodedata
from datetime import UTC, date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

SERPER_URL = "https://google.serper.dev/news"
DEFAULT_OUTPUT_JSON = "data/fetch_news_companies.json"


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
DEFAULT_TICKERS = [
    "PTSC",
    "PV Drilling",
    "PV GAS",
    "BSR",
    "Petrolimex",
    "PV OIL",
    "PV POW",
    "Điện lực Dầu khí Nhơn Trạch 2",
    "REE",
    "Tập đoàn Hà Đô",
    "PC1",
    "Điện Gia Lai GEG",
    "EVNGENCO3",
    "Tư vấn Xây dựng Điện 2",
]

TICKER_CODES = {
    "PTSC": "PVS",
    "PV Drilling": "PVD",
    "PV GAS": "GAS",
    "BSR": "BSR",
    "Petrolimex": "PLX",
    "PV OIL": "OIL",
    "PV POW": "POW",
    "Điện lực Dầu khí Nhơn Trạch 2": "NT2",
    "REE": "REE",
    "Tập đoàn Hà Đô": "HDG",
    "PC1": "PC1",
    "Điện Gia Lai GEG": "GEG",
    "EVNGenco3 PGV": "PGV",
    "Tư vấn Xây dựng Điện 2": "TV2",
}

# Sector / theme keywords searched as their own queries alongside the tickers,
# to widen coverage of industry news not tied to a single company.
DEFAULT_KEYWORDS = [
    "xăng dầu",
    "dầu khí",
    "dự án điện",
    "quy hoạch điện",
    "tư vấn điện",
    "xây lắp điện",
    "cơ chế DPPA",
    "điện khí LNG",
    "Lô B",
    "Nhơn Trạch 3",
    "Nhơn Trạch 4",
    "VinEnergo",
    "năng lượng Trung Nam",
    "PVN",
    "EVN",
    "giá bán điện",
    "giá điện",
    "điều chỉnh giá điện",
    "khung giá điện",
    "xăng sinh học",
    "năng lượng tái tạo",
]

# If company keyword is not mentioned in title/snippet, article must contain
# at least one keyword from this list (mapped by ticker code) to pass filter.
MUST_HAVE_KEYWORDS_BY_CODE: dict[str, list[str]] = {
    "PVS": ["dầu", "khí"],
    "PVD": ["dầu", "khí"],
    "GAS": ["khí"],
    "BSR": ["dầu"],
    "PLX": ["xăng", "dầu"],
    "OIL": ["xăng", "dầu"],
    "POW": ["điện"],
    "NT2": ["điện"],
    "PC1": ["điện"],
    "GEG": ["điện"],
    "PGV": ["điện"],
}

# Keyword groups and weights for importance scoring from an equity research view.
IMPORTANT_TOPICS: dict[str, dict[str, Any]] = {
    "earnings": {
        "weight": 4,
        "keywords": [
            "earnings",
            "quarterly results",
            "annual results",
            "profit",
            "net income",
            "revenue",
            "ebitda",
            "lntt",
            "lnst",
            "ket qua kinh doanh",
            "bao cao tai chinh",
            "loi nhuan",
            "doanh thu",
            "lai sau thue",
            "lai rong",
            "vuot ke hoach",
            "giai trinh",
        ],
    },
    "guidance_plan": {
        "weight": 3,
        "keywords": [
            "guidance",
            "outlook",
            "target",
            "plan",
            "forecast",
            "ke hoach",
            "muc tieu",
            "du bao",
            "dai hoi dong co dong",
            "dhdcd",
            "nghi quyet",
            "ke hoach kinh doanh",
        ],
    },
    "dividend_capital": {
        "weight": 4,
        "keywords": [
            "dividend",
            "cash dividend",
            "stock dividend",
            "rights issue",
            "private placement",
            "capital increase",
            "share issuance",
            "buyback",
            "co tuc",
            "phat hanh",
            "tang von",
            "mua lai co phieu",
            "tra co tuc",
            "chia co tuc",
            "phat hanh co phieu",
            "ngay dang ky cuoi cung",
        ],
    },
    "mna_deals": {
        "weight": 4,
        "keywords": [
            "m&a",
            "merger",
            "acquisition",
            "deal",
            "strategic partnership",
            "joint venture",
            "thoai von",
            "mua lai",
            "hop tac chien luoc",
        ],
    },
    "project_contract": {
        "weight": 3,
        "keywords": [
            "contract",
            "signed",
            "tender",
            "epc",
            "fsoe",
            "project",
            "cod",
            "commissioning",
            "ppa",
            "du an",
            "goi thau",
            "ky hop dong",
            "van hanh thuong mai",
            "khoi cong",
            "khanh thanh",
            "hoa luoi",
            "phat dien",
            "ky ket",
        ],
    },
    "operations_risk": {
        "weight": 3,
        "keywords": [
            "shutdown",
            "outage",
            "incident",
            "delay",
            "cost overrun",
            "suspension",
            "tam dung",
            "su co",
            "cham tien do",
            "doi von",
        ],
    },
    "regulatory_legal": {
        "weight": 3,
        "keywords": [
            "regulation",
            "circular",
            "decree",
            "investigation",
            "lawsuit",
            "penalty",
            "tax",
            "thanh tra",
            "xu phat",
            "nghi dinh",
            "thong tu",
        ],
    },
    "broker_views": {
        "weight": 2,
        "keywords": [
            "target price",
            "upgrade",
            "downgrade",
            "recommendation",
            "valuation",
            "analyst",
            "gia muc tieu",
            "khuyen nghi",
            "dinh gia",
        ],
    },
    "financing_debt": {
        "weight": 3,
        "keywords": [
            "bond",
            "loan",
            "refinancing",
            "interest expense",
            "debt",
            "trai phieu",
            "vay no",
            "tai co cau no",
            "vay von",
            "tin dung",
            "giai ngan",
            "bao lanh",
        ],
    },
    "leadership": {
        "weight": 2,
        "keywords": [
            "appoint",
            "resignation",
            "ceo",
            "chairman",
            "board of directors",
            "bo nhiem",
            "mien nhiem",
            "tu nhiem",
            "thay doi nhan su",
            "tong giam doc",
            "chu tich hdqt",
            "thanh vien hdqt",
        ],
    },
}

# Daily commodity-price tracker articles are not price-moving for these stocks;
# matching one pushes the score well below the importance threshold.
NOISE_PATTERNS = [
    "gia gas hom nay",
    "gia gas ban le",
    "gia gas ngay",
    "gia gas thang",
    "gia gas trong nuoc",
    "gia gas the gioi",
    "gia ga hom nay",
    "gia xang hom nay",
    "gia xang dau hom nay",
    "gas prices today",
]


def date_to_serper(d: date) -> str:
    return f"{d.month}/{d.day}/{d.year}"


def iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


@lru_cache(maxsize=4096)
def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    no_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    # NFKD does not decompose the Vietnamese "đ", so fold it to "d" explicitly;
    # otherwise accent-free keywords like "dien" never match "điện" in the text.
    folded = no_accents.lower().replace("đ", "d")
    return " ".join(folded.split())


# The topic keywords are fixed; normalizing them per article (classify_article
# runs once per raw result) repeated ~110 unicodedata passes for every article.
# Pre-normalize once at import into (topic, weight, normalized_keywords) tuples.
_TOPIC_MATCHERS: list[tuple[str, int, tuple[str, ...]]] = [
    (topic, int(config["weight"]), tuple(normalize_text(k) for k in config["keywords"]))
    for topic, config in IMPORTANT_TOPICS.items()
]


# Vietnam is UTC+7 year-round (no DST), so a fixed offset avoids depending on
# the IANA tz database (absent on Windows without the `tzdata` package).
VN_TZ = timezone(timedelta(hours=7))

# Serper "date" strings (hl=vi) after accent-folding via normalize_text, e.g.
# "9 gio truoc", "45 phut truoc", "2 ngay truoc", or absolute "5 thg 7, 2026".
_REL_RE = re.compile(r"(\d+)\s*(giay|phut|gio|ngay|tuan|thang|nam)\s+truoc")
_ABS_RE = re.compile(r"(\d{1,2})\s*thg\s*(\d{1,2})(?:,?\s*(\d{4}))?")
# unit -> timedelta kwarg. thang/nam handled separately (no exact timedelta).
_REL_UNITS: dict[str, str] = {
    "giay": "seconds",
    "phut": "minutes",
    "gio": "hours",
    "ngay": "days",
    "tuan": "weeks",
}


def resolve_news_date(
    serper_date: object,
    run_at: datetime | None,
    query_date_iso: str,
) -> str:
    """Derive the article's publish date as a VN calendar day ("YYYY-MM-DD").

    Primary source is Serper's relative ``date`` ("X gio truoc"): since we know
    exactly when the query ran (``run_at``), publish_instant = run_at - age. An
    elapsed duration is timezone-independent, so this avoids the cdr timezone skew
    that makes ``query_date`` a day early for early-morning-VN articles. Falls
    back to ``query_date_iso`` when ``date`` is missing/unparseable.
    """
    if run_at is None:
        return query_date_iso

    text = normalize_text(str(serper_date or ""))
    run_vn = run_at.astimezone(VN_TZ)

    match = _REL_RE.search(text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        if unit in _REL_UNITS:
            publish = run_at - timedelta(**{_REL_UNITS[unit]: amount})
            return publish.astimezone(VN_TZ).date().isoformat()
        # Months/years have no exact timedelta; approximate.
        days = 30 * amount if unit == "thang" else 365 * amount
        publish = run_at - timedelta(days=days)
        return publish.astimezone(VN_TZ).date().isoformat()

    if "hom qua" in text:
        return (run_vn.date() - timedelta(days=1)).isoformat()
    if "hom nay" in text or "vua" in text:  # "vua xong" = just now
        return run_vn.date().isoformat()

    absolute = _ABS_RE.search(text)
    if absolute:
        day = int(absolute.group(1))
        month = int(absolute.group(2))
        year = int(absolute.group(3)) if absolute.group(3) else run_vn.year
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            pass

    return query_date_iso


def build_query(ticker: str) -> str:
    # Unquoted (broad match) like a normal Google News search — an exact-phrase
    # "dự án điện" is far narrower and misses relevant coverage.
    return ticker


def classify_article(
    article: dict[str, Any],
    ticker: str,
    ticker_code: str,
    query_date: date,
    run_at: datetime | None = None,
    threshold: int = 3,
) -> dict[str, Any]:
    raw_title = str(article.get("title", "")).lower()
    raw_snippet = str(article.get("snippet", "")).lower()
    raw_haystack = f"{raw_title} {raw_snippet}"
    title = normalize_text(str(article.get("title", "")))
    snippet = normalize_text(str(article.get("snippet", "")))
    source = normalize_text(str(article.get("source", "")))
    haystack = f"{title} {snippet}"

    matched_topics: list[str] = []
    score = 0

    for topic, weight, keywords in _TOPIC_MATCHERS:
        if any(k in haystack for k in keywords):
            matched_topics.append(topic)
            score += weight

    has_company_mention = normalize_text(ticker) in haystack
    if has_company_mention:
        score += 2
    else:
        score -= 1

    must_have_keywords = MUST_HAVE_KEYWORDS_BY_CODE.get(ticker_code, [])
    has_must_have_keyword = False
    if must_have_keywords:
        # Must-have keywords are accent-sensitive and matched on raw title/snippet.
        has_must_have_keyword = any(k.lower() in raw_haystack for k in must_have_keywords)

    # If ticker has no must-have list (n/a), require explicit company mention.
    if has_company_mention:
        passes_must_have = True
    elif must_have_keywords:
        passes_must_have = has_must_have_keyword
    else:
        passes_must_have = False

    if any(s in source for s in ["cafef", "vietstock", "ndh", "tinnhanhchungkhoan", "reuters", "bloomberg"]):
        score += 1

    is_noise = any(pattern in haystack for pattern in NOISE_PATTERNS)
    if is_noise:
        score -= 5

    # Drop image payload to keep JSON compact and analyst-friendly.
    classified = {k: v for k, v in article.items() if k != "imageUrl"}
    classified["importance_score"] = score
    classified["is_noise"] = is_noise
    classified["matched_topics"] = matched_topics
    classified["must_have_keywords"] = must_have_keywords
    classified["has_must_have_keyword"] = has_must_have_keyword
    classified["passes_must_have"] = passes_must_have
    # Analyst/price relevance: must be on-topic (a real corporate/financial event),
    # pass the company/sector gate, clear the score threshold, and not be price-tracker noise.
    classified["is_important"] = (
        bool(matched_topics) and passes_must_have and score >= threshold and not is_noise
    )
    classified["ticker"] = ticker_code
    classified["ticker_name"] = ticker
    classified["query_date"] = query_date.isoformat()
    # Publish date derived from Serper's relative "date" anchored to the actual
    # query time; query_date is only the fallback when "date" can't be parsed.
    classified["news_date"] = resolve_news_date(
        article.get("date"), run_at, classified["query_date"]
    )
    classified["has_company_mention"] = has_company_mention
    return classified


def search_news_by_day(
    session: requests.Session,
    api_key: str,
    query: str,
    target_date: date,
    lang: str,
    country: str,
    num: int,
    timeout: int = 30,
) -> dict[str, Any]:
    """Fetch up to ``num`` news for one day (absolute date range, relevance order).

    Serper caps a single call at ~10 results, so we PAGE (``page:1,2,…``) to
    collect up to ``num``.
    """
    date_str = date_to_serper(target_date)
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    tbs = f"cdr:1,cd_min:{date_str},cd_max:{date_str}"

    collected: list[dict[str, Any]] = []
    pages = max(1, (num + 9) // 10)  # 10 -> 1 page, 30 -> 3 pages
    for page in range(1, pages + 1):
        payload = {"q": query, "tbs": tbs, "hl": lang, "gl": country, "num": 10, "page": page}
        response = session.post(SERPER_URL, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        batch = response.json().get("news", [])
        if not batch:
            break
        collected.extend(batch)
        if len(batch) < 10:
            break
        if page < pages:
            time.sleep(0.4)  # gentle pacing to avoid Serper rate-limiting
    return {"news": collected[:num]}


def collect_news(
    api_key: str,
    tickers: list[str],
    start_date: date,
    end_date: date,
    lang: str,
    country: str,
    num: int,
    sleep_seconds: float,
    importance_threshold: int,
    keywords: list[str] | None = None,
) -> dict[str, Any]:
    session = requests.Session()

    total_articles = 0
    total_calls = 0

    # Search the tickers and the sector keywords with the same per-day flow.
    # Keywords carry the "THEME" code so the must-have filter falls back to
    # requiring the keyword itself to appear in the article.
    search_terms = [(t, TICKER_CODES.get(t, t), "ticker") for t in tickers]
    search_terms += [(k, "THEME", "keyword") for k in (keywords or [])]

    all_articles: list[dict[str, Any]] = []
    for d in iter_dates(start_date, end_date):
        for term, term_code, kind in search_terms:
            query = build_query(term)
            total_calls += 1
            try:
                data = search_news_by_day(
                    session=session,
                    api_key=api_key,
                    query=query,
                    target_date=d,
                    lang=lang,
                    country=country,
                    num=num,
                )
            except requests.RequestException:
                continue
            # Timestamp the response so resolve_news_date can turn Serper's
            # relative "X gio truoc" into an absolute publish date.
            query_at = datetime.now(UTC)

            raw_articles = data.get("news", [])
            for a in raw_articles:
                classified = classify_article(
                    a,
                    term,
                    ticker_code=term_code,
                    query_date=d,
                    run_at=query_at,
                    threshold=importance_threshold,
                )
                # High-recall mode: keep every article relevant to the search term
                # (mentions the company or carries the required sector keyword). We
                # no longer hard-drop on is_important, so material news that falls
                # outside the topic taxonomy (e.g. criminal/legal cases) isn't missed;
                # importance_score / is_important are retained for ranking downstream.
                if classified["passes_must_have"]:
                    all_articles.append(classified)
            total_articles += len(raw_articles)

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    news = flatten_articles(all_articles)

    return {
        "metadata": {
            "collected_at": datetime.now(UTC).isoformat(),
            "since": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "latest_date": latest_news_date(news),
            "tickers": tickers,
            "keywords": keywords or [],
            "language": lang,
            "country": country,
            "num_per_query": num,
            "importance_threshold": importance_threshold,
            "total_api_calls": total_calls,
            "total_news": len(news),
        },
        "news": news,
    }


def latest_news_date(news: list[dict[str, Any]]) -> str | None:
    dates = [str(item.get("news_date", "")) for item in news if item.get("news_date")]
    return max(dates) if dates else None


def flatten_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-(term, day) classifications into one object per article.

    Articles are deduped by link; the search terms that surfaced the article are
    unioned into a `search` array, and their ticker codes into a `ticker` array
    (theme-keyword hits carry the "THEME" code and are skipped). Each object keeps
    title, snippet, source, link, search, ticker, news_date, importance_score and
    the ranking flags is_important / matched_topics / is_noise.
    """
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for article in articles:
        link = str(article.get("link", "")).strip()
        term = str(article.get("ticker_name", "") or article.get("ticker", ""))
        # ticker_code from classify_article; "THEME" marks a sector-keyword hit
        # (not a company), so it never becomes a ticker.
        code = str(article.get("ticker", "")).strip()
        code = code if code and code != "THEME" else ""
        key = link or f"{term}|{article.get('title', '')}"
        if key not in by_key:
            by_key[key] = {
                "title": article.get("title", ""),
                "snippet": article.get("snippet", ""),
                "source": article.get("source", ""),
                "link": link,
                "search": [term] if term else [],
                "ticker": [code] if code else [],
                # Publish date from resolve_news_date (Serper "date" anchored to the
                # query time); query_date is the fallback when that can't be parsed.
                "news_date": article.get("news_date", "") or article.get("query_date", ""),
                "importance_score": int(article.get("importance_score", 0)),
                # Ranking/filtering signals kept for the frontend now that we no
                # longer hard-drop: is_important flags analyst-grade events, is_noise
                # flags daily price-tracker clutter, matched_topics explains the score.
                "is_important": bool(article.get("is_important")),
                "is_noise": bool(article.get("is_noise")),
                "matched_topics": list(article.get("matched_topics") or []),
            }
            order.append(key)
            continue

        merged = by_key[key]
        if term and term not in merged["search"]:
            merged["search"].append(term)
        if code and code not in merged["ticker"]:
            merged["ticker"].append(code)
        merged["importance_score"] = max(merged["importance_score"], int(article.get("importance_score", 0)))
        merged["is_important"] = merged.get("is_important", False) or bool(article.get("is_important"))
        merged["is_noise"] = merged.get("is_noise", False) or bool(article.get("is_noise"))
        for topic in article.get("matched_topics") or []:
            if topic not in merged["matched_topics"]:
                merged["matched_topics"].append(topic)
        if not merged["snippet"] and article.get("snippet"):
            merged["snippet"] = article.get("snippet")
        # First real date wins; only fill when still empty (never reshuffle).
        if not merged.get("news_date") and article.get("news_date"):
            merged["news_date"] = article.get("news_date")

    news = [by_key[key] for key in order]
    news.sort(
        key=lambda item: (str(item.get("news_date", "")), item.get("importance_score", 0)),
        reverse=True,
    )
    return news


def save_json(data: dict[str, Any], output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json_if_exists(file_path: str) -> dict[str, Any] | None:
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _news_key(item: dict[str, Any]) -> str:
    link = str(item.get("link", "")).strip()
    return link or f"{','.join(item.get('search', []))}|{item.get('title', '')}"


def merge_payloads(existing: dict[str, Any], new_data: dict[str, Any]) -> dict[str, Any]:
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in existing.get("news", []):
        item.setdefault("search", [])
        item.setdefault("ticker", [])
        item.setdefault("matched_topics", [])
        key = _news_key(item)
        if key not in by_key:
            by_key[key] = item
            order.append(key)

    for item in new_data.get("news", []):
        key = _news_key(item)
        if key in by_key:
            merged_item = by_key[key]
            for term in item.get("search", []):
                if term not in merged_item.get("search", []):
                    merged_item.setdefault("search", []).append(term)
            for code in item.get("ticker", []):
                if code not in merged_item.get("ticker", []):
                    merged_item.setdefault("ticker", []).append(code)
            merged_item["importance_score"] = max(
                int(merged_item.get("importance_score", 0)), int(item.get("importance_score", 0))
            )
            merged_item["is_important"] = bool(merged_item.get("is_important")) or bool(item.get("is_important"))
            merged_item["is_noise"] = bool(merged_item.get("is_noise")) or bool(item.get("is_noise"))
            for topic in item.get("matched_topics", []):
                if topic not in merged_item.get("matched_topics", []):
                    merged_item.setdefault("matched_topics", []).append(topic)
            if not merged_item.get("snippet") and item.get("snippet"):
                merged_item["snippet"] = item.get("snippet")
            # Keep the already-stored date stable across re-runs; only fill an
            # empty one so updates never reshuffle previously-shown dates.
            if not merged_item.get("news_date") and item.get("news_date"):
                merged_item["news_date"] = item.get("news_date")
        else:
            by_key[key] = item
            order.append(key)

    merged_news = [by_key[key] for key in order]
    merged_news.sort(
        key=lambda item: (str(item.get("news_date", "")), item.get("importance_score", 0)),
        reverse=True,
    )

    old_meta = existing.get("metadata", {})
    new_meta = new_data.get("metadata", {})
    merged_meta = dict(old_meta)
    merged_meta["collected_at"] = datetime.now(UTC).isoformat()
    merged_meta["since"] = old_meta.get("since", new_meta.get("since"))
    merged_meta["end_date"] = new_meta.get("end_date", old_meta.get("end_date"))
    merged_meta["latest_date"] = latest_news_date(merged_news)
    merged_meta["tickers"] = new_meta.get("tickers", old_meta.get("tickers", []))
    merged_meta["keywords"] = new_meta.get("keywords", old_meta.get("keywords", []))
    merged_meta["language"] = new_meta.get("language", old_meta.get("language", "vi"))
    merged_meta["country"] = new_meta.get("country", old_meta.get("country", "vn"))
    merged_meta["num_per_query"] = new_meta.get("num_per_query", old_meta.get("num_per_query", 10))
    merged_meta["importance_threshold"] = new_meta.get("importance_threshold", old_meta.get("importance_threshold", 3))
    merged_meta["total_api_calls"] = int(old_meta.get("total_api_calls", 0)) + int(new_meta.get("total_api_calls", 0))
    merged_meta["total_news"] = len(merged_news)

    return {"metadata": merged_meta, "news": merged_news}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect daily coverage news from Google Serper and save to JSON.")
    parser.add_argument(
        "--since",
        default="2026-07-01",
        help="Cutoff date YYYY-MM-DD for the FIRST run; end is always today. Later runs "
        "continue from the saved latest_date to today (default: 2026-05-01).",
    )
    parser.add_argument("--lang", default="vi", help="Language code (default: vi)")
    parser.add_argument("--country", default="vn", help="Country code (default: vn)")
    parser.add_argument("--num", type=int, default=30, help="Max news results per query, collected via paging in blocks of 10 (default: 30 = up to 3 pages)")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between API calls (default: 0.1)")
    parser.add_argument("--importance-threshold", type=int, default=3, help="Minimum score to mark important (default: 3)")
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore existing output and rebuild the file from the provided date range.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_JSON,
        help="Output JSON path",
    )
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated ticker list",
    )
    parser.add_argument(
        "--keywords",
        default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated sector/theme keywords searched alongside the tickers",
    )
    parser.add_argument(
        "--api-key",
        default=load_secret("SERPER_API_KEY"),
        help="Serper API key (default: SERPER_API_KEY env var, then .secrets.toml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Set --api-key or SERPER_API_KEY environment variable.")

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise ValueError("Ticker list is empty")

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    existing = None if args.full_refresh else load_json_if_exists(args.output)

    # Date range = [cutoff .. today]. First run uses --since; later runs continue
    # from the latest date already saved (re-pulling that day to catch additions).
    saved_latest = existing.get("metadata", {}).get("latest_date") if isinstance(existing, dict) else None
    since_text = saved_latest or args.since
    start_date = datetime.strptime(since_text, "%Y-%m-%d").date()
    end_date = date.today()

    if start_date > end_date:
        print(f"No new day to update. Latest output already up to {start_date}.")
        return

    print(
        f"Collecting news from {start_date} to {end_date} for "
        f"{len(tickers)} tickers and {len(keywords)} keywords..."
    )
    payload = collect_news(
        api_key=args.api_key,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        lang=args.lang,
        country=args.country,
        num=args.num,
        sleep_seconds=args.sleep,
        importance_threshold=args.importance_threshold,
        keywords=keywords,
    )
    final_payload = merge_payloads(existing, payload) if existing else payload
    save_json(final_payload, args.output)

    meta = final_payload["metadata"]
    print("Done.")
    print(f"Output: {args.output}")
    print(f"API calls: {meta['total_api_calls']}")
    print(f"Total news objects: {meta['total_news']}")


if __name__ == "__main__":
    main()
