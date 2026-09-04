"""Fetch investor-relations (IR) documents — disclosures, reports, AGM materials —
for the covered tickers. Companion to releases_processor.py but for the IR/document
sections of each company site. Each record keeps only: title, snippet (if any),
date, url. Document files (PDFs) are never downloaded; the url is the link as it
appears on the listing page.
"""

import argparse
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


@dataclass
class IRDocument:
	title: str
	snippet: str
	date: str
	url: str
	ticker: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Text / date helpers
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tiff")


def is_image_url(url: str) -> bool:
	"""True if the URL points at an image file (ignoring any query string)."""
	path = urlsplit(url).path.lower() if url else ""
	return path.endswith(IMAGE_EXTENSIONS)


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def strip_accents(text: str) -> str:
	normalized = unicodedata.normalize("NFKD", text)
	folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
	return folded.replace("đ", "d").replace("Đ", "D")


def extract_date(text: str) -> str:
	"""Find a dd/mm/yyyy or dd.mm.yyyy date in text and return it as dd/mm/yyyy."""
	match = re.search(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", text)
	if match is None:
		return ""
	day, month, year = match.groups()
	return f"{int(day):02d}/{int(month):02d}/{year}"


def iso_to_ddmmyyyy(raw_date: str) -> str:
	parts = raw_date[:10].split("-")
	if len(parts) != 3:
		return ""
	year, month, day = parts
	return f"{day}/{month}/{year}"


def parse_date_tuple(date_str: str) -> tuple[int, int, int] | None:
	match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str or "")
	if match is None:
		return None
	day, month, year = (int(g) for g in match.groups())
	return (year, month, day)


def is_recent(date_str: str, since: tuple[int, int, int]) -> bool:
	"""True if the date is on/after the cutoff. Unknown dates are kept."""
	parsed = parse_date_tuple(date_str)
	if parsed is None:
		return True
	return parsed >= since


VIETNAMESE_MONTHS = {
	"gieng": 1,
	"mot": 1,
	"hai": 2,
	"ba": 3,
	"tu": 4,
	"nam": 5,
	"sau": 6,
	"bay": 7,
	"tam": 8,
	"chin": 9,
	"muoi": 10,
	"muoi mot": 11,
	"muoi hai": 12,
}


def parse_vietnamese_month_date(raw_date: str) -> str:
	"""Parse "[Đăng ngày:] DD Tháng <tên tháng> YYYY" into dd/mm/yyyy."""
	normalized = strip_accents(normalize_text(raw_date)).lower()
	normalized = normalized.replace("dang ngay", " ").replace(":", " ")
	normalized = re.sub(r"\s+", " ", normalized).strip()

	match = re.search(r"(\d{1,2})\s+thang\s+([a-z ]+?)\s+(\d{4})", normalized)
	if match is None:
		return extract_date(raw_date)

	day = int(match.group(1))
	month = VIETNAMESE_MONTHS.get(match.group(2).strip())
	year = int(match.group(3))
	if month is None:
		return extract_date(raw_date)
	return f"{day:02d}/{month:02d}/{year:04d}"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def get_session() -> requests.Session:
	session = requests.Session()
	session.verify = False
	session.headers.update(
		{
			"User-Agent": (
				"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
				"AppleWebKit/537.36 (KHTML, like Gecko) "
				"Chrome/125.0.0.0 Safari/537.36"
			),
			"Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
		}
	)
	return session


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str:
	response = session.get(url, timeout=timeout)
	response.raise_for_status()
	return response.text


def deduplicate(items: Iterable[IRDocument]) -> list[IRDocument]:
	seen: set[str] = set()
	unique: list[IRDocument] = []
	for item in items:
		key = item.url or f"{item.date}|{item.title}"
		if key in seen:
			continue
		seen.add(key)
		unique.append(item)
	return unique


def paginate(
	session: requests.Session,
	make_url: Callable[[int], str],
	parse_fn: Callable[[str, str], list[IRDocument]],
	base_url: str,
	max_pages: int,
	since: tuple[int, int, int],
	delay_seconds: float = 0.3,
	single_page: bool = False,
) -> list[IRDocument]:
	items: list[IRDocument] = []
	pages = 1 if single_page else max(1, max_pages)
	for page in range(1, pages + 1):
		try:
			html = fetch_html(session, make_url(page))
		except requests.RequestException:
			break
		page_items = parse_fn(html, base_url)
		if not page_items:
			break
		items.extend(item for item in page_items if is_recent(item.date, since))
		# Listings are newest-first: once a whole page is older than the cutoff, stop.
		if not single_page and not any(
			(parsed := parse_date_tuple(item.date)) is not None and parsed >= since
			for item in page_items
		):
			break
		if delay_seconds > 0 and page < pages:
			time.sleep(delay_seconds)
	return items


# ---------------------------------------------------------------------------
# Per-ticker parsers (listing pages only)
# ---------------------------------------------------------------------------

def parse_bsr(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for row in soup.select("tr.document-item"):
		title_node = row.select_one("p.document-title")
		if title_node is None:
			continue
		title = normalize_text(title_node.get_text(" ", strip=True))
		cells = row.find_all("td")
		date = extract_date(cells[1].get_text(" ", strip=True)) if len(cells) > 1 else ""
		link = row.select_one("td a[download]") or row.select_one("td a[href]")
		url = urljoin(base_url, str(link.get("href", ""))) if link else ""
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_gas_disclosures(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for article in soup.select("div.EDN_simpleArticle"):
		link = article.select_one("h3.simpleArticleTitle a")
		if link is None:
			continue
		title = normalize_text(link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date_node = article.select_one("span.EDN_simpleDate")
		date = parse_vietnamese_month_date(date_node.get_text(" ", strip=True)) if date_node else ""
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_gas_documents(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for article in soup.select("article"):
		link = article.select_one(".col-sm-9 a[href]")
		date_node = article.select_one(".col-sm-2 time")
		if link is None or date_node is None:
			continue
		title = normalize_text(link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date = parse_vietnamese_month_date(date_node.get_text(" ", strip=True))
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_hdg(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for row in soup.select("table.table tbody tr"):
		link = row.select_one("a.sedt")
		if link is None:
			continue
		title = normalize_text(str(link.get("title", "")) or link.get_text(" ", strip=True))
		date = extract_date(row.get_text(" ", strip=True))
		doc_id = str(link.get("data-id", "")).strip()
		# The row link is javascript:void(id); the document is served from this
		# deterministic endpoint (no PDF download performed here).
		url = f"{base_url}/Ajax/Content/ViewDetail?id={doc_id}" if doc_id else ""
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_nt2(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	table = soup.find("table")
	if table is None:
		return items
	for row in table.find_all("tr")[1:]:
		cells = row.find_all("td")
		if not cells:
			continue
		link = cells[0].find("a", href=True)
		if link is None:
			continue
		title = normalize_text(link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date = ""
		if len(cells) > 1:
			time_node = cells[1].find("time")
			date = extract_date((time_node or cells[1]).get_text(" ", strip=True))
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_pgv(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for item in soup.select("div.sn-new-item"):
		title_node = item.select_one("p.title")
		link = item.select_one("a.detail[href]") or item.select_one("a[href][target='_blank']")
		if title_node is None or link is None:
			continue
		title = normalize_text(title_node.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date_node = item.select_one("span.time")
		date = extract_date(date_node.get_text(" ", strip=True)) if date_node else ""
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_plx(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for post in soup.select(".post-detail-list .post-default"):
		link = post.select_one(".post-default__title a[href]")
		if link is None:
			continue
		title = normalize_text(link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date = ""
		meta_node = post.select_one(".post-default__meta")
		if meta_node is not None:
			date = extract_date(meta_node.get_text(" ", strip=True))
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_pvd(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for row in soup.select("table tbody tr"):
		cells = row.find_all("td")
		if len(cells) < 3:
			continue
		link = cells[1].find("a", href=True)
		if link is None:
			continue
		title = normalize_text(link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date = extract_date(cells[2].get_text(" ", strip=True))
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_pvs(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for card in soup.select(".news-card-item"):
		link = card.select_one(".title a[href]")
		if link is None:
			continue
		title = normalize_text(link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date_node = card.select_one(".time time") or card.select_one("time")
		date = extract_date(date_node.get_text(" ", strip=True)) if date_node else ""
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


def parse_ree(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	for row in soup.select("div.vii-report-item"):
		title_node = row.select_one("h3.vii-report-item__title")
		if title_node is None:
			continue
		title = normalize_text(title_node.get_text(" ", strip=True))
		link = row.select_one("a.vii-report-item__download-icon[href]")
		url = urljoin(base_url, str(link.get("href", ""))) if link else ""
		date_node = row.select_one("div.vii-report-item__col.date time") or row.select_one("time")
		date = ""
		if date_node is not None:
			date = extract_date(date_node.get("datetime", "")) or extract_date(date_node.get_text(" ", strip=True))
			if not date:
				date = iso_to_ddmmyyyy(str(date_node.get("datetime", "")))
		items.append(IRDocument(title=title, snippet="", date=date, url=url))
	return items


# ---------------------------------------------------------------------------
# Per-ticker fetchers
# ---------------------------------------------------------------------------

def parse_pow(html: str, base_url: str) -> list[IRDocument]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[IRDocument] = []
	# Featured item uses h2.title, grid items use h3.title; both share .title a.
	for post in soup.select(".list-post .post-item"):
		link = post.select_one(".title a[href]")
		if link is None:
			continue
		title = normalize_text(str(link.get("title", "")) or link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))
		date_node = post.select_one("span.published-date")
		date = extract_date(date_node.get_text(" ", strip=True)) if date_node else ""
		meta_node = post.select_one(".meta")
		snippet = normalize_text(meta_node.get_text(" ", strip=True)) if meta_node else ""
		items.append(IRDocument(title=title, snippet=snippet, date=date, url=url))
	return items


def fetch_bsr(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	# All rows are present in one server response; the site paginates client-side.
	sections = [
		"/tai-lieu-danh-cho-nha-dau-tu",
		"/cong-bo-thong-tin-khac",
		"/dai-hoi-co-dong",
		"/bao-cao",
	]
	items: list[IRDocument] = []
	for path in sections:
		items += paginate(session, lambda p, u=base_url + path: u, parse_bsr, base_url, 1, since, single_page=True)
	return items


def fetch_gas(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	qhcd = "/quan-he-co-%C4%91ong"
	items: list[IRDocument] = []
	# Disclosure feed: paginated via pgrid/574/pageid/{N}.
	items += paginate(
		session,
		lambda p: base_url + qhcd if p == 1 else f"{base_url}{qhcd}/pgrid/574/pageid/{p}",
		parse_gas_disclosures,
		base_url,
		max_pages,
		since,
	)
	# Document library: the first page of each section is server-rendered.
	items += paginate(
		session,
		lambda p: f"{base_url}{qhcd}/tai-lieu-co-%C4%91ong",
		parse_gas_documents,
		base_url,
		1,
		since,
		single_page=True,
	)
	return items


def fetch_geg(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	api = "https://web-be.geccom.vn/api/v2/front/post"
	sections = [
		f"{api}/tai-lieu-bao-cao/posts?limit=50&page={{p}}",
		f"{api}/cong-bo-thong-tin/posts?categories=QHCD_00022&limit=50&page={{p}}",
		f"{api}/dai-hoi-dong-co-dong/posts?categories=agm-thuong-nien&limit=50&page={{p}}",
	]
	items: list[IRDocument] = []
	for template in sections:
		for page in range(1, max(1, max_pages) + 1):
			try:
				response = session.get(template.format(p=page), timeout=30)
				response.raise_for_status()
				payload = response.json()
			except (requests.RequestException, ValueError):
				break
			records = payload.get("data") if isinstance(payload, dict) else payload
			if not records:
				break
			page_dates: list[str] = []
			for record in records:
				title = normalize_text(str(record.get("title", "")))
				if not title:
					continue
				date = iso_to_ddmmyyyy(str(record.get("published_start") or record.get("created_at") or ""))
				page_dates.append(date)
				if not is_recent(date, since):
					continue
				snippet = normalize_text(str(record.get("short_description") or ""))
				slug = str(record.get("slug") or "").lstrip("/")
				# Only the article slug becomes the URL; never fall back to the
				# featured image, so image files never leak into the output.
				url = f"{base_url}/{slug}" if slug else ""
				items.append(IRDocument(title=title, snippet=snippet, date=date, url=url))
			# Newest-first: stop this section once a whole page predates the cutoff.
			if not any((p := parse_date_tuple(d)) is not None and p >= since for d in page_dates):
				break
	return items


def fetch_hdg(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	# Page 1 is server-rendered; later pages come from an AJAX endpoint that
	# returns the same table markup.
	items: list[IRDocument] = []
	for page in range(1, max(1, max_pages) + 1):
		url = (
			f"{base_url}/quan-he-co-dong?t=19"
			if page == 1
			else f"{base_url}/Ajax/Home/LoadContentShareHolders?id=19&p={page}"
		)
		try:
			html = fetch_html(session, url)
		except requests.RequestException:
			break
		page_items = parse_hdg(html, base_url)
		if not page_items:
			break
		items.extend(item for item in page_items if is_recent(item.date, since))
		if not any((p := parse_date_tuple(item.date)) is not None and p >= since for item in page_items):
			break
		if page < max(1, max_pages):
			time.sleep(0.3)
	return items


def fetch_nt2(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	sections = [
		"/quan-he-co-dong/cong-bo-thong-tin",
		"/quan-he-co-dong/tong-ket-va-bao-cao",
	]
	items: list[IRDocument] = []
	for path in sections:
		items += paginate(
			session,
			lambda p, u=base_url + path: f"{u}?pagenumber={p}",
			parse_nt2,
			base_url,
			max_pages,
			since,
		)
	return items


def fetch_pgv(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	return paginate(
		session,
		lambda p: f"{base_url}/quan-he-nha-dau-tu.html",
		parse_pgv,
		base_url,
		1,
		since,
		single_page=True,
	)


def fetch_plx(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	return paginate(
		session,
		lambda p: (
			f"{base_url}/ndi/thong_tin_co_dong.html"
			if p == 1
			else f"{base_url}/ndi/thong_tin_co_dong/{p}.html"
		),
		parse_plx,
		base_url,
		max_pages,
		since,
	)


def fetch_pvd(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	sections = [
		"/quan-he-co-dong/thong-tin-cong-bo",
		"/quan-he-co-dong/dai-hoi-dong-co-dong",
	]
	items: list[IRDocument] = []
	for path in sections:
		items += paginate(
			session,
			lambda p, u=base_url + path: f"{u}?pagenumber={p}",
			parse_pvd,
			base_url,
			max_pages,
			since,
		)
	return items


def fetch_pvs(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	return paginate(
		session,
		lambda p: f"{base_url}/co-dong/tin-co-dong?pagenumber={p}",
		parse_pvs,
		base_url,
		max_pages,
		since,
	)


def fetch_ree(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	return paginate(
		session,
		lambda p: (
			f"{base_url}/danh-muc-tai-lieu/cong-bo-thong-tin/"
			if p == 1
			else f"{base_url}/danh-muc-tai-lieu/cong-bo-thong-tin/page/{p}/"
		),
		parse_ree,
		base_url,
		max_pages,
		since,
	)


def fetch_pow(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	tags = [
		"thong-tintai-lieu-co-dong-20",
		"dai-hoi-co-dong-23",
		"bao-cao-tai-chinh-10",
	]
	items: list[IRDocument] = []
	for tag in tags:
		items += paginate(
			session,
			lambda p, t=tag: f"{base_url}/vi/tag/{t}/page-{p}",
			parse_pow,
			base_url,
			max_pages,
			since,
		)
	return items


# PECC2 serves its IR library via an AJAX module: the listing page holds only the
# category/year controls, and each (category, year) pair is loaded by POSTing to
# load_item.html, which returns a JSON list of documents. The year is passed as a
# site-internal index (not the calendar year), so the index map is read from the
# page's year dropdown.
PECC2_IR_PATH = "/vn/quan-he-dau-tu.html"
PECC2_AJAX_PATH = "/vn/quan-he-dau-tu/ajax/load_item.html"
# Category ids: 4 Công bố thông tin, 5 Quản trị công ty, 6 Điều lệ, 7 Báo cáo,
# 8 Báo cáo quản trị/tài chính, 9 ĐHĐCĐ, 11 ĐHĐCĐ thường niên, 13 Báo cáo thường niên.
PECC2_CATEGORIES = [4, 5, 6, 7, 8, 9, 11, 13]


def parse_pecc2_date(txt_date_post: str) -> str:
	"""Parse the split badge markup "<span>DD</span>MM/YYYY" into dd/mm/yyyy."""
	match = re.search(r"<span>\s*(\d{1,2})\s*</span>\s*(\d{1,2})/(\d{4})", txt_date_post)
	if match is None:
		return extract_date(txt_date_post)
	day, month, year = match.groups()
	return f"{int(day):02d}/{int(month):02d}/{year}"


def pecc2_year_index_map(session: requests.Session, base_url: str) -> dict[int, int]:
	"""Map calendar year -> the site's internal year index used by load_item."""
	try:
		html = fetch_html(session, urljoin(base_url, PECC2_IR_PATH))
	except requests.RequestException:
		return {}
	soup = BeautifulSoup(html, "html.parser")
	mapping: dict[int, int] = {}
	for link in soup.select(".cat_year a[data-year]"):
		year_match = re.search(r"(\d{4})", link.get_text(" ", strip=True))
		if year_match is None:
			continue
		mapping[int(year_match.group(1))] = int(str(link.get("data-year")))
	return mapping


def fetch_pecc2(session: requests.Session, base_url: str, max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	year_map = pecc2_year_index_map(session, base_url)
	if not year_map:
		return []
	# Documents are partitioned by year on the site; query every year on/after the
	# cutoff. If the cutoff predates the site's coverage, fall back to all years.
	target_indexes = sorted({idx for year, idx in year_map.items() if year >= since[0]}, reverse=True)
	if not target_indexes:
		target_indexes = sorted(set(year_map.values()), reverse=True)

	ajax_url = urljoin(base_url, PECC2_AJAX_PATH)
	headers = {"X-Requested-With": "XMLHttpRequest"}
	items: list[IRDocument] = []
	for cat_id in PECC2_CATEGORIES:
		for year_index in target_indexes:
			try:
				response = session.post(
					ajax_url,
					data={"cat_id": cat_id, "year": year_index},
					headers=headers,
					timeout=30,
				)
				response.raise_for_status()
				response.encoding = "utf-8"
				payload = response.json()
			except (requests.RequestException, ValueError):
				continue
			records = payload.get("items") if isinstance(payload, dict) else None
			if not records:
				continue
			for record in records:
				title = normalize_text(str(record.get("title", "")))
				url = str(record.get("link", "")).strip()
				if not title or not url:
					continue
				date = parse_pecc2_date(str(record.get("txt_date_post", "")))
				items.append(IRDocument(title=title, snippet="", date=date, url=url))
			time.sleep(0.3)
	return items


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCES: dict[str, dict] = {
	"bsr": {"base_url": "https://bsr.com.vn", "fetch": fetch_bsr},
	"gas": {"base_url": "https://www.pvgas.com.vn", "fetch": fetch_gas},
	"geg": {"base_url": "https://geccom.vn", "fetch": fetch_geg},
	"hdg": {"base_url": "https://hado.com.vn", "fetch": fetch_hdg},
	"nt2": {"base_url": "https://www.pvpnt2.vn", "fetch": fetch_nt2},
	"pgv": {"base_url": "https://www.genco3.com", "fetch": fetch_pgv},
	"pow": {"base_url": "https://pvpower.vn", "fetch": fetch_pow},
	"plx": {"base_url": "https://www.petrolimex.com.vn", "fetch": fetch_plx},
	"pvd": {"base_url": "https://www.pvdrilling.com.vn", "fetch": fetch_pvd},
	"pvs": {"base_url": "https://www.ptsc.com.vn", "fetch": fetch_pvs},
	"ree": {"base_url": "https://www.reecorp.com", "fetch": fetch_ree},
	"pecc2": {"base_url": "https://pecc2.com", "fetch": fetch_pecc2},
}


DEFAULT_OUTPUT = Path("data") / "fetch_documents_companies.json"
ALL_SOURCES = list(SOURCES.keys())
DEFAULT_SINCE = "2026-05-01"


def latest_date_str(docs: list[dict]) -> str | None:
	best: tuple[int, int, int] | None = None
	for item in docs:
		parsed = parse_date_tuple(str(item.get("date", "")))
		if parsed is not None and (best is None or parsed > best):
			best = parsed
	if best is None:
		return None
	return f"{best[0]:04d}-{best[1]:02d}-{best[2]:02d}"


def _doc_key(item: dict) -> str:
	return str(item.get("url") or f"{item.get('date', '')}|{item.get('title', '')}")


def merge_documents(existing: list[dict], new_items: list[IRDocument]) -> list[dict]:
	"""Merge new documents into existing, deduping by url (or date|title) and
	unioning the ticker arrays of duplicates."""
	by_key: dict[str, dict] = {}
	order: list[str] = []
	for record in existing:
		record.setdefault("ticker", [])
		key = _doc_key(record)
		if key not in by_key:
			by_key[key] = record
			order.append(key)

	for item in new_items:
		record = asdict(item)
		key = _doc_key(record)
		if key in by_key:
			existing_record = by_key[key]
			tickers = list(existing_record.get("ticker") or [])
			for code in record.get("ticker", []):
				if code not in tickers:
					tickers.append(code)
			existing_record["ticker"] = tickers
			if not existing_record.get("snippet") and record.get("snippet"):
				existing_record["snippet"] = record["snippet"]
		else:
			by_key[key] = record
			order.append(key)

	merged = [by_key[key] for key in order]
	merged.sort(key=lambda record: parse_date_tuple(str(record.get("date", ""))) or (0, 0, 0), reverse=True)
	return merged


def load_existing(path: Path) -> dict | None:
	if not path.exists():
		return None
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except (ValueError, OSError):
		return None
	return data if isinstance(data, dict) else None


def save_payload(payload: dict, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_all(sources: list[str], max_pages: int, since: tuple[int, int, int]) -> list[IRDocument]:
	aggregated: list[IRDocument] = []
	for source in sources:
		code = source.upper()
		try:
			items = fetch_documents(source=source, max_pages=max_pages, since=since)
		except requests.RequestException as exc:
			print(f"  [{code}] error: {exc}")
			continue
		for item in items:
			item.ticker = [code]
		aggregated.extend(items)
		print(f"  [{code}] {len(items)} document(s) since cutoff")
	return aggregated


def fetch_documents(
	source: str,
	max_pages: int = 30,
	since: tuple[int, int, int] = (2026, 5, 1),
) -> list[IRDocument]:
	config = SOURCES[source]
	with get_session() as session:
		items = config["fetch"](session, config["base_url"], max_pages, since)
	# Drop any record whose URL is a bare image file so images never enter the output.
	return [
		item
		for item in deduplicate(items)
		if is_recent(item.date, since) and not is_image_url(item.url)
	]


def parse_since(value: str) -> tuple[int, int, int]:
	# Accept both the CLI/ISO form (YYYY-MM-DD) and the dd/mm/yyyy the document
	# feed saves as its metadata latest_date. A stray dd/mm/yyyy used to crash the
	# whole json pipeline at step 1, blocking the news/press/upload steps behind it.
	value = value.strip()
	if "/" in value:
		day, month, year = (int(part) for part in value.split("/"))
	else:
		year, month, day = (int(part) for part in value.split("-"))
	return (year, month, day)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fetch IR documents from all company sources into one JSON file, "
		"appending only items newer than the last update."
	)
	parser.add_argument(
		"--sources",
		default="all",
		help=f"Comma-separated sources to fetch, or 'all' (default). Available: {', '.join(ALL_SOURCES)}.",
	)
	parser.add_argument(
		"--pages",
		type=int,
		default=30,
		help="Safety cap on pages for paginated sources; paging stops early once a page predates the cutoff (default: 30).",
	)
	parser.add_argument(
		"--since",
		default=DEFAULT_SINCE,
		help=f"Cutoff date YYYY-MM-DD for the FIRST run (default: {DEFAULT_SINCE}). "
		"On later runs the cutoff is the latest date already saved in the output file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
	)
	parser.add_argument(
		"--full-refresh",
		action="store_true",
		help="Ignore the existing output and rebuild from --since.",
	)
	parser.add_argument("--print", action="store_true", help="Print merged rows to stdout.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	output_path = args.output

	if args.sources.strip().lower() == "all":
		sources = ALL_SOURCES
	else:
		sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
		unknown = [s for s in sources if s not in SOURCES]
		if unknown:
			raise ValueError(f"Unknown source(s): {', '.join(unknown)}. Available: {', '.join(ALL_SOURCES)}")

	existing = None if args.full_refresh else load_existing(output_path)
	existing_docs = existing.get("documents", []) if existing else []

	# Incremental: on later runs, only fetch from the latest date already saved.
	saved_latest = existing.get("metadata", {}).get("latest_date") if existing else None
	since = parse_since(saved_latest) if saved_latest else parse_since(args.since)

	print(f"Fetching {len(sources)} source(s) since {since[2]:02d}/{since[1]:02d}/{since[0]} ...")
	new_items = collect_all(sources, max_pages=max(1, args.pages), since=since)
	merged_docs = merge_documents(existing_docs, new_items)

	added = len(merged_docs) - len(existing_docs)
	payload = {
		"metadata": {
			"last_updated": datetime.now().isoformat(timespec="seconds"),
			"latest_date": latest_date_str(merged_docs),
			"sources": sources,
			"total_documents": len(merged_docs),
		},
		"documents": merged_docs,
	}
	save_payload(payload, output_path)

	print(f"Saved {len(merged_docs)} documents ({added} new) to {output_path}.")
	if args.print:
		for record in merged_docs:
			print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
	main()
