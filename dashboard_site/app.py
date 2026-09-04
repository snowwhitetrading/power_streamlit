"""Power dashboard website — FastAPI backend.

Implements the seven tabs of `README_mongodb.md`:

    1 Government   2 Weather   3 Cost   4 Price   5 Volume   6 Company   7 News

Tabs 1–6 read `dc_commodity.ThiTruongDien` (filtered by `dataset`); Tab 7 and the
per-company news block read `dc_commodity.TinNganhDien`. The one exception is the
four `fact_*` reference tables of Tab 1, which are read from their source CSVs —
see the docstring of `facts.py`.

Run (from the project root):
    .venv/Scripts/python.exe -m uvicorn dashboard_site.app:app --reload --port 8100
then open http://127.0.0.1:8100
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import facts, queries
from .data import FREQS, NEWS_COLLECTION, charts, clear_cache, get_db, parse_date

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"

app = FastAPI(title="Thị trường điện — Dashboard")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _freq(value: str, allowed=FREQS) -> str:
    if value not in allowed:
        raise HTTPException(400, f"freq phải là một trong {list(allowed)}")
    return value


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# ==========================================================================
# meta
# ==========================================================================
@app.get("/api/meta")
def api_meta():
    """Dataset inventory — row counts and the latest date the pipeline uploaded."""
    rows = charts().aggregate([
        {"$group": {"_id": "$dataset", "n": {"$sum": 1},
                    "latest": {"$max": "$date"}, "updated": {"$max": "$updated_at"}}},
        {"$sort": {"_id": 1}},
    ])
    datasets = [
        {
            "dataset": r["_id"],
            "docs": r["n"],
            "latest": r["latest"].strftime("%Y-%m-%d") if r.get("latest") else None,
            "updated_at": r["updated"].strftime("%Y-%m-%d %H:%M") if r.get("updated") else None,
        }
        for r in rows
    ]
    return {"datasets": datasets, "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")}


@app.post("/api/cache/clear")
def api_cache_clear():
    return {"cleared": clear_cache()}


# ==========================================================================
# Tab 1 — Government
# ==========================================================================
@app.get("/api/gov/pdp8-capacity")
def api_pdp8_capacity():
    return facts.pdp8_capacity()


@app.get("/api/gov/pdp8-capex")
def api_pdp8_capex():
    return facts.pdp8_capex()


@app.get("/api/gov/price-framework")
def api_price_framework():
    return facts.price_framework()


@app.get("/api/gov/pdp8-plants")
def api_pdp8_plants():
    return facts.pdp8_plants()


# ==========================================================================
# Tab 2 — Weather
# ==========================================================================
@app.get("/api/weather/enso-probability")
def api_enso_probability():
    return queries.enso_probability()


@app.get("/api/weather/enso-condition")
def api_enso_condition(date_from: str = Query(""), date_to: str = Query("")):
    return queries.enso_condition(parse_date(date_from), parse_date(date_to))


@app.get("/api/weather/temperature")
def api_temperature(freq: str = Query("month")):
    return queries.temperature(_freq(freq))


# ==========================================================================
# Tab 3 — Cost
# ==========================================================================
@app.get("/api/cost/coal")
def api_coal_cost():
    return queries.coal_cost()


@app.get("/api/cost/gas")
def api_gas_cost():
    return queries.gas_cost()


@app.get("/api/cost/reservoir-yoy")
def api_reservoir_yoy(freq: str = Query("month")):
    return queries.reservoir_yoy(_freq(freq, ("month", "quarter", "year")))


# ==========================================================================
# Tab 4 — Price
# ==========================================================================
@app.get("/api/price/cgm")
def api_cgm(freq: str = Query("month")):
    return queries.cgm_price(_freq(freq))


@app.get("/api/price/can")
def api_can():
    return queries.can_price()


@app.get("/api/price/retail")
def api_retail():
    return queries.retail_price()


# ==========================================================================
# Tab 5 — Volume
# ==========================================================================
@app.get("/api/volume/total")
def api_volume_total(freq: str = Query("month")):
    return queries.volume_total(_freq(freq))


@app.get("/api/volume/breakdown")
def api_volume_breakdown(freq: str = Query("month")):
    return queries.volume_breakdown(_freq(freq))


@app.get("/api/volume/mismatch")
def api_volume_mismatch(freq: str = Query("month")):
    return queries.capacity_mismatch(_freq(freq))


@app.get("/api/volume/capacity-installed")
def api_capacity_installed():
    return queries.capacity_installed()


@app.get("/api/volume/capacity-region")
def api_capacity_region(freq: str = Query("month")):
    return queries.capacity_region(_freq(freq))


# ==========================================================================
# Tab 6 — Company
# ==========================================================================
@app.get("/api/company/list")
def api_company_list():
    return {
        "companies": [
            {"ticker": t, "name": m["name"], "dataset": m["dataset"], "cadence": m["cadence"]}
            for t, m in queries.COMPANIES.items()
        ]
    }


@app.get("/api/company/{ticker}")
def api_company(ticker: str):
    data = queries.company(ticker)
    if not data:
        raise HTTPException(404, f"Không có dữ liệu cho mã {ticker.upper()}")
    return data


@app.get("/api/company/{ticker}/reservoirs")
def api_company_reservoirs(ticker: str):
    if ticker.upper() not in queries.COMPANIES:
        raise HTTPException(404, f"Không có mã {ticker.upper()}")
    return queries.company_reservoirs(ticker)


@app.get("/api/company/{ticker}/news")
def api_company_news(
    ticker: str,
    important_only: bool = Query(False),
    limit: int = Query(20, ge=1, le=100),
):
    """News for one company — the three feeds unioned via the per-company key map.

    `document`/`press` are tagged with `tickers[]`; `news` is keyword-driven and
    matched on `search[]` instead (README, Tab 6).
    """
    keys = queries.COMPANY_NEWS_KEYS.get(ticker.upper())
    if not keys:
        raise HTTPException(404, f"Không có mã {ticker.upper()}")
    news_branch: dict = {"type": "news", "search": {"$in": keys["search"]}}
    if important_only:
        news_branch["is_important"] = True
    else:
        news_branch["is_noise"] = {"$ne": True}
    query = {"$or": [
        {"type": {"$in": ["document", "press"]}, "tickers": {"$in": keys["tickers"]}},
        news_branch,
    ]}
    cursor = (get_db()[NEWS_COLLECTION].find(query)
              .sort([("date", -1), ("importance_score", -1)])
              .limit(limit))
    return {"items": [_news_item(d) for d in cursor], "keys": keys}


def _news_item(doc: dict) -> dict:
    date = doc.get("date")
    return {
        "type": doc.get("type"),
        "title": doc.get("title"),
        "url": doc.get("url"),
        "source": doc.get("source"),
        "snippet": doc.get("snippet"),
        "date": date.strftime("%Y-%m-%d") if date else (doc.get("date_text") or ""),
        "tags": doc.get("tickers") or doc.get("search") or [],
        "is_important": bool(doc.get("is_important")),
        "importance": doc.get("importance_score"),
    }


# ==========================================================================
# Tab 7 — News
# --------------------------------------------------------------------------
# The standalone news site already implements the feed (filtering, tag
# canonicalisation, relevance gate). Re-register its handlers here instead of
# duplicating them, so both sites stay in step.
# ==========================================================================
try:
    from news_site.app import api_news as _api_news, api_tags as _api_tags

    app.get("/api/news")(_api_news)
    app.get("/api/tags")(_api_tags)
    NEWS_TAB = True
except Exception:  # pragma: no cover - the dashboard still works without it
    NEWS_TAB = False

    @app.get("/api/news")
    def api_news_unavailable():
        return JSONResponse(
            {"items": [], "total": 0,
             "error": "news_site không khả dụng — chạy từ thư mục gốc dự án."},
            status_code=503,
        )
