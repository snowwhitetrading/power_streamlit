import argparse
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)


SOURCES = {
	"pvd": {
		"base_url": "https://www.pvdrilling.com.vn",
		"list_path": "/tin-tuc/tin-pv-drilling",
		"output": Path("data") / "coverage_news_pvd.json",
	},
	"pvs": {
		"base_url": "https://www.ptsc.com.vn",
		"list_path": "/tin-tuc/tin-ptsc-1/san-xuat-kinh-doanh",
		"output": Path("data") / "coverage_news_pvs.json",
	},
	"gas": {
		"base_url": "https://www.pvgas.com.vn",
		"list_path": "/tin-tuc/category/tin-hoat-dong",
		"page_url_template": "https://www.pvgas.com.vn/tin-tuc/pgrid/588/pageid/{page}/pid/588/evl/0/categoryid/63/categoryname/tin-hoat-dong",
		"output": Path("data") / "coverage_news_gas.json",
	},
	"bsr": {
		"base_url": "https://bsr.com.vn",
		"list_path": "/vi/tin-tuc-su-kien",
		"page_url_template": "https://bsr.com.vn/vi/tin-tuc-su-kien?p_p_id=com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_wrvs&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_wrvs_delta=7&p_r_p_resetCur=false&_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_wrvs_cur={page}",
		"output": Path("data") / "coverage_news_bsr.json",
	},
	"nt2": {
		"base_url": "https://www.pvpnt2.vn",
		"list_path": "/tin-tuc-hoat-dong/tin-tuc-su-kien",
		"page_url_template": "https://www.pvpnt2.vn/tin-tuc-hoat-dong/tin-tuc-su-kien?pagenumber={page}",
		"output": Path("data") / "coverage_news_nt2.json",
	},
	"hdg": {
		"base_url": "https://hado.com.vn",
		"list_path": "/tin-du-an",
		"page_url_template": "https://hado.com.vn/tin-du-an/page/{page}",
		"output": Path("data") / "coverage_news_hdg.json",
	},
	"geg": {
		"base_url": "https://geccom.vn",
		"list_path": "/thong-cao-bao-chi/thong-cao-bao-chi-ir",
		"page_url_template": "https://geccom.vn/thong-cao-bao-chi/thong-cao-bao-chi-ir?page={page}",
		"output": Path("data") / "coverage_news_geg.json",
	},
	"pgv": {
		"base_url": "https://www.genco3.com",
		"list_path": "/thong-bao/thong-bao-evngenco-3.html",
		"page_url_template": "https://www.genco3.com/thong-bao/thong-bao-evngenco-3.html",
		"output": Path("data") / "coverage_news_pgv.json",
	},
	"pow": {
		"base_url": "https://pvpower.vn",
		"list_path": "/vi/tag/hoat-dong-pv-power-5",
		"page_url_template": "https://pvpower.vn/vi/tag/hoat-dong-pv-power-5/page-{page}",
		"output": Path("data") / "coverage_news_pow.json",
	},
	"ree": {
		"base_url": "https://www.reecorp.com",
		"list_path": "/tin-tuc-su-kien",
		"page_url_template": "https://www.reecorp.com/tin-tuc-su-kien/page/{page}/",
		"output": Path("data") / "coverage_news_ree.json",
	},
	"plx": {
		"base_url": "https://www.petrolimex.com.vn",
		"list_path": "/ndi/thong-cao-bao-chi/1.html",
		# Petrolimex spreads IR-relevant news across two listings; both are merged
		# into one PLX coverage file (overlapping cross-posts are deduplicated).
		# "tintuc-sukien" is the broad tin-tuc/su-kien feed (widest, carries the
		# newest items); "thong-cao-bao-chi" is the formal press-release listing.
		"page_url_templates": [
			"https://www.petrolimex.com.vn/ndi/thong-cao-bao-chi/{page}.html",
			"https://www.petrolimex.com.vn/tintuc-sukien/{page}.html",
		],
		"output": Path("data") / "coverage_news_plx.json",
	},
	"pecc2": {
		"base_url": "https://pecc2.com",
		"list_path": "/vn/thong-tin-su-kien.html/p-1",
		"page_url_template": "https://pecc2.com/vn/thong-tin-su-kien.html/p-{page}",
		"output": Path("data") / "coverage_news_pecc2.json",
	},
}


@dataclass
class NewsCandidate:
	title: str
	snippet: str
	date: str
	url: str


@dataclass
class NewsItem:
	title: str
	snippet: str
	date: str
	url: str
	ticker: list[str] = field(default_factory=list)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".tiff")


def is_image_url(url: str) -> bool:
	"""True if the URL points at an image file (ignoring any query string)."""
	path = urlsplit(url).path.lower() if url else ""
	return path.endswith(IMAGE_EXTENSIONS)


def normalize_text(text: str) -> str:
	return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def normalize_date(raw_date: str) -> str:
	compact = normalize_text(raw_date)
	compact = re.sub(r"\s*/\s*", "/", compact)
	compact = re.sub(r"\s*\.\s*", "/", compact)
	return compact


def strip_accents(text: str) -> str:
	normalized = unicodedata.normalize("NFKD", text)
	return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def parse_vietnamese_month_date(raw_date: str) -> str:
	cleaned = normalize_text(raw_date)
	if not cleaned:
		return ""

	# Already in dd/mm/yyyy or dd.mm.yyyy style.
	if re.search(r"\d{1,2}\s*[./]\s*\d{1,2}\s*[./]\s*\d{4}", cleaned):
		return normalize_date(cleaned)

	normalized = strip_accents(cleaned).lower()
	normalized = re.sub(r"\s+", " ", normalized)
	normalized = normalized.replace("thang", " thang ")
	normalized = re.sub(r"\s+", " ", normalized).strip()

	month_map = {
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

	match = re.search(r"(\d{1,2})\s+thang\s+([a-z ]+)\s+(\d{4})", normalized)
	if match is None:
		return normalize_date(cleaned)

	day = int(match.group(1))
	month_key = re.sub(r"\s+", " ", match.group(2)).strip()
	year = int(match.group(3))
	month = month_map.get(month_key)
	if month is None:
		return normalize_date(cleaned)

	return f"{day:02d}/{month:02d}/{year:04d}"


def parse_pvd_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	for article in soup.select(".main-wrap .item"):
		title_node = article.select_one(".title h3")
		snippet_node = article.select_one(".brief-content")
		date_node = article.select_one(".date time")
		link_node = article.select_one("a[href]")

		if title_node is None:
			continue

		title = normalize_text(title_node.get_text(" ", strip=True))
		snippet = ""
		if snippet_node is not None:
			snippet = normalize_text(snippet_node.get_text(" ", strip=True))

		date = ""
		if date_node is not None:
			date = normalize_date(date_node.get_text(" ", strip=True))

		url = ""
		if link_node is not None and link_node.get("href"):
			url = urljoin(base_url, str(link_node.get("href")))

		items.append(NewsCandidate(title=title, snippet=snippet, date=date, url=url))

	return items


def parse_pvs_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	for article in soup.select("section.news-list.news-list-1.section-bot .news-card-item"):
		title_link = article.select_one(".title a[href]")
		date_node = article.select_one("time")

		if title_link is None:
			continue

		title = normalize_text(title_link.get_text(" ", strip=True))
		date = ""
		if date_node is not None:
			date = normalize_date(date_node.get_text(" ", strip=True))

		url = urljoin(base_url, str(title_link.get("href", "")))
		items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	return items


def parse_gas_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	for article in soup.select(".edn_588_article_list_wrapper article.edn_article"):
		title_link = article.select_one(".edn_articleTitle a[href]")
		snippet_node = article.select_one(".edn_articleSummary")
		date_node = article.select_one("time")

		if title_link is None:
			continue

		title = normalize_text(title_link.get_text(" ", strip=True))
		snippet = ""
		if snippet_node is not None:
			snippet = normalize_text(snippet_node.get_text(" ", strip=True))

		date = ""
		if date_node is not None:
			date = parse_vietnamese_month_date(date_node.get_text(" ", strip=True))

		url = urljoin(base_url, str(title_link.get("href", "")))
		items.append(NewsCandidate(title=title, snippet=snippet, date=date, url=url))

	return items


def extract_onclick_url(onclick_value: str, base_url: str) -> str:
	match = re.search(r"location\.href\s*=\s*['\"]([^'\"]+)['\"]", onclick_value)
	if match is None:
		return ""
	return urljoin(base_url, match.group(1))


def extract_date_from_container(container) -> str:
	if container is None:
		return ""

	date_candidates = [
		normalize_date(node.get_text(" ", strip=True))
		for node in container.select("p")
	]
	for text in reversed(date_candidates):
		if re.search(r"\d{1,2}/\d{1,2}/\d{4}", text):
			return text

	if date_candidates:
		return date_candidates[-1]
	return ""


def parse_bsr_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	main_article = soup.select_one(".tintucsukien-main-section .last-news")
	if main_article is not None:
		title_node = main_article.select_one(".last-news-title")
		snippet_node = main_article.select_one(".last-news-des")
		date = extract_date_from_container(main_article.select_one(".last-news-time"))
		onclick_value = str(main_article.get("onclick", ""))
		url = extract_onclick_url(onclick_value, base_url)

		if title_node is not None:
			title = normalize_text(title_node.get_text(" ", strip=True))
			snippet = ""
			if snippet_node is not None:
				snippet = normalize_text(snippet_node.get_text(" ", strip=True))
			items.append(NewsCandidate(title=title, snippet=snippet, date=date, url=url))

	for article in soup.select(".tintucsukien-main-section .news-section .news-item"):
		title_node = article.select_one(".news-item-title")
		snippet_node = article.select_one(".news-item-des")
		date = extract_date_from_container(article.select_one(".news-item-time"))

		clickable_node = article.select_one(".content-ttsk-right[onclick]")
		onclick_value = ""
		if clickable_node is not None:
			onclick_value = str(clickable_node.get("onclick", ""))
		url = extract_onclick_url(onclick_value, base_url)

		if title_node is None:
			continue

		title = normalize_text(title_node.get_text(" ", strip=True))
		snippet = ""
		if snippet_node is not None:
			snippet = normalize_text(snippet_node.get_text(" ", strip=True))

		items.append(NewsCandidate(title=title, snippet=snippet, date=date, url=url))

	return items


def parse_nt2_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	# Featured article block on top-left.
	featured = soup.select_one(".news-wrap .news-top .news-first")
	if featured is not None:
		link_node = featured.select_one("h2 a[href]")
		date_node = featured.select_one("time")
		if link_node is not None:
			title = normalize_text(link_node.get_text(" ", strip=True))
			date = ""
			if date_node is not None:
				date = normalize_date(date_node.get_text(" ", strip=True))
			url = urljoin(base_url, str(link_node.get("href", "")))
			items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	# Right-side top compact articles.
	for article in soup.select(".news-wrap .news-top .news-top-right section.row"):
		link_node = article.select_one("h3 a[href]")
		date_node = article.select_one("time")
		if link_node is None:
			continue
		title = normalize_text(link_node.get_text(" ", strip=True))
		date = ""
		if date_node is not None:
			date = normalize_date(date_node.get_text(" ", strip=True))
		url = urljoin(base_url, str(link_node.get("href", "")))
		items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	# Bottom list articles.
	for article in soup.select(".news-wrap .news-bottom .news-item"):
		link_node = article.select_one(".news-desc h3 a[href]")
		date_node = article.select_one(".news-desc time")
		if link_node is None:
			continue
		title = normalize_text(link_node.get_text(" ", strip=True))
		date = ""
		if date_node is not None:
			date = normalize_date(date_node.get_text(" ", strip=True))
		url = urljoin(base_url, str(link_node.get("href", "")))
		items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	return items


def parse_hdg_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	for article in soup.select(".content-new.pj-header .list-item2 .item"):
		title_link = article.select_one(".desc a[href]")
		date_node = article.select_one(".desc span")
		if title_link is None:
			continue

		title = normalize_text(title_link.get_text(" ", strip=True))
		date = ""
		if date_node is not None:
			raw_date = normalize_text(date_node.get_text(" ", strip=True))
			date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", raw_date)
			if date_match is not None:
				date = normalize_date(date_match.group(0))

		url = urljoin(base_url, str(title_link.get("href", "")))
		items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	return items


# geccom.vn is a Next.js app: the IR list is embedded as escaped JSON inside the
# server-rendered React Flight payload rather than as plain HTML, so we extract the
# article records with regex instead of BeautifulSoup. The ?page query is handled
# client-side, so every page returns the same recent batch of IR releases.
GEG_POST_PATTERN = re.compile(
	r'\\"slug\\":\\"(thong-cao-bao-chi/[^\\"]+)\\",'
	r'\\"locale\\":\\"vi\\",\\"is_active\\":1,'
	r'\\"created_by\\":\\"[^\\"]*\\",'
	r'\\"collections\\":\[\],\\"templates\\":\[\],'
	r'\\"position\\":\d+,'
	r'\\"title\\":\\"([^\\"]+)\\",'
	r'\\"type\\":\\"thong-cao-bao-chi\\"'
)
GEG_PUBLISHED_PATTERN = re.compile(r'\\"published_start\\":\\"([0-9T:.\-Z]+)\\"')
GEG_SHORT_DESC_PATTERN = re.compile(r'\\"short_description\\":\\"([^\\"]*)\\"')


def iso_date_to_ddmmyyyy(raw_date: str) -> str:
	parts = raw_date[:10].split("-")
	if len(parts) != 3:
		return normalize_date(raw_date)
	year, month, day = parts
	return f"{day}/{month}/{year}"


def parse_geg_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	post_matches = list(GEG_POST_PATTERN.finditer(html))

	# A few records carry an article-specific short_description nearby; map it by slug
	# so we can attach a snippet to those releases that expose one.
	desc_by_slug: dict[str, str] = {}
	for index, match in enumerate(post_matches):
		block_end = (
			post_matches[index + 1].start()
			if index + 1 < len(post_matches)
			else len(html)
		)
		window = html[match.end():min(block_end, match.end() + 4000)]
		desc_match = GEG_SHORT_DESC_PATTERN.search(window)
		if desc_match is not None and desc_match.group(1):
			desc_by_slug.setdefault(match.group(1), normalize_text(desc_match.group(1)))

	items: list[NewsCandidate] = []
	seen_slugs: set[str] = set()
	for match in post_matches:
		slug = match.group(1)
		if slug.endswith("landing-page") or slug in seen_slugs:
			continue

		tail = html[match.end():match.end() + 2500]
		# Keep only releases that belong to the "Thông cáo báo chí IR" category.
		if "thong-cao-bao-chi-ir" not in tail:
			continue

		published_match = GEG_PUBLISHED_PATTERN.search(tail)
		if published_match is None:
			continue

		seen_slugs.add(slug)
		title = normalize_text(match.group(2))
		date = iso_date_to_ddmmyyyy(published_match.group(1))
		url = urljoin(base_url, "/" + slug)
		items.append(
			NewsCandidate(
				title=title,
				snippet=desc_by_slug.get(slug, ""),
				date=date,
				url=url,
			)
		)

	return items


def parse_pgv_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	# The top banner duplicates the newest list item, so only the .nw-item list is
	# parsed. Detail pages expose no usable description, so snippets stay empty.
	for item in soup.select(".nw-list .nw-item"):
		link = item.select_one(".content-title a[href]")
		if link is None:
			continue

		title = normalize_text(str(link.get("title", "")) or link.get_text(" ", strip=True))

		date = ""
		time_node = item.select_one(".blog-time .time")
		if time_node is not None:
			date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", time_node.get_text(" ", strip=True))
			if date_match is not None:
				date = normalize_date(date_match.group(0))

		url = urljoin(base_url, str(link.get("href", "")))
		items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	return items


def parse_ree_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	# Both the pinned "Tin nổi bật" highlights and the dated archive grid use the
	# same .vii-blog-item block. Highlights carry no date on the list page, so the
	# date (and snippet) for every item is enriched from its detail page later.
	for block in soup.select(".vii-blog-item"):
		link = block.select_one("a.vii-blog-item__inner[href]")
		title_node = block.select_one(".vii-blog-item__title")
		if link is None or title_node is None:
			continue

		title = normalize_text(title_node.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))

		date = ""
		time_node = block.select_one("time")
		if time_node is not None:
			date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", time_node.get_text(" ", strip=True))
			if date_match is not None:
				date = normalize_date(date_match.group(0))

		items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	return items


def fetch_ree_detail(
	session: requests.Session,
	article_url: str,
	timeout: int = 30,
) -> tuple[str, str]:
	"""Return (date, snippet) for a REE article from its detail page metadata."""
	if not article_url:
		return "", ""

	response = session.get(article_url, timeout=timeout)
	response.raise_for_status()
	soup = BeautifulSoup(response.text, "html.parser")

	date = ""
	published = soup.select_one("meta[property='article:published_time']")
	if published is not None and published.get("content"):
		date = iso_date_to_ddmmyyyy(str(published.get("content")))

	snippet = ""
	for selector in ["meta[property='og:description']", "meta[name='description']"]:
		node = soup.select_one(selector)
		if node is None:
			continue
		content = normalize_text(str(node.get("content", "")))
		if content:
			snippet = content
			break

	return date, snippet


# Petrolimex publishes a recurring fuel price-adjustment notice every few days
# ("Petrolimex điều chỉnh giá xăng dầu từ ... ngày ..."). These are excluded so the
# PLX feed keeps only substantive corporate/IR news. strip_accents leaves the
# Vietnamese "đ" intact, so it is folded to "d" before matching.
PLX_PRICE_NEWS_PATTERN = re.compile(r"dieu chinh gia xang")


def is_fuel_price_news(title: str) -> bool:
	normalized = strip_accents(title).lower().replace("đ", "d")
	return bool(PLX_PRICE_NEWS_PATTERN.search(normalized))


def parse_plx_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	for post in soup.select(".post-detail-list .post-default"):
		title_link = post.select_one(".post-default__title a[href]")
		if title_link is None:
			continue

		title = normalize_text(title_link.get_text(" ", strip=True))
		if is_fuel_price_news(title):
			continue

		date = ""
		meta_node = post.select_one(".post-default__meta")
		if meta_node is not None:
			date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", meta_node.get_text(" ", strip=True))
			if date_match is not None:
				date = normalize_date(date_match.group(0))

		url = urljoin(base_url, str(title_link.get("href", "")))
		items.append(NewsCandidate(title=title, snippet="", date=date, url=url))

	return items


def fetch_plx_page(
	session: requests.Session,
	page: int,
	timeout: int = 30,
) -> list[NewsCandidate]:
	base_url = str(SOURCES["plx"]["base_url"])
	items: list[NewsCandidate] = []
	for template in SOURCES["plx"]["page_url_templates"]:
		response = session.get(template.format(page=page), timeout=timeout)
		response.raise_for_status()
		items.extend(parse_plx_news_items(response.text, base_url))
	return items


def parse_pow_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	# Featured item uses h2.title, grid items use h3.title; both share .title a.
	for post in soup.select(".list-post .post-item"):
		link = post.select_one(".title a[href]")
		if link is None:
			continue

		title = normalize_text(str(link.get("title", "")) or link.get_text(" ", strip=True))
		url = urljoin(base_url, str(link.get("href", "")))

		date = ""
		date_node = post.select_one("span.published-date")
		if date_node is not None:
			date_match = re.search(r"(\d{1,2})[/.](\d{1,2})[/.](\d{4})", date_node.get_text(" ", strip=True))
			if date_match is not None:
				date = f"{int(date_match.group(1)):02d}/{int(date_match.group(2)):02d}/{date_match.group(3)}"

		snippet = ""
		meta_node = post.select_one(".meta")
		if meta_node is not None:
			snippet = normalize_text(meta_node.get_text(" ", strip=True))

		items.append(NewsCandidate(title=title, snippet=snippet, date=date, url=url))

	return items


def parse_pecc2_news_items(html: str, base_url: str) -> list[NewsCandidate]:
	soup = BeautifulSoup(html, "html.parser")
	items: list[NewsCandidate] = []

	for article in soup.select("div.newsItem"):
		link = article.select_one(".titleH a[href]") or article.select_one("a[href]")
		if link is None:
			continue

		title = normalize_text(str(link.get("title", "")) or link.get_text(" ", strip=True))

		date = ""
		date_node = article.select_one(".dateSty")
		if date_node is not None:
			date_match = re.search(r"\d{1,2}/\d{1,2}/\d{4}", date_node.get_text(" ", strip=True))
			if date_match is not None:
				date = normalize_date(date_match.group(0))

		snippet = ""
		snippet_node = article.select_one(".tendN")
		if snippet_node is not None:
			snippet = normalize_text(snippet_node.get_text(" ", strip=True))

		url = urljoin(base_url, str(link.get("href", "")))
		items.append(NewsCandidate(title=title, snippet=snippet, date=date, url=url))

	return items


def build_page_url(source: str, page: int) -> str:
	source_cfg = SOURCES[source]
	page_template = source_cfg.get("page_url_template")
	if isinstance(page_template, str):
		return page_template.format(page=page)
	return f"{source_cfg['base_url']}{source_cfg['list_path']}?pagenumber={page}"


def get_session() -> requests.Session:
	session = requests.Session()
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


def fetch_page(
	session: requests.Session,
	source: str,
	page: int,
	timeout: int = 30,
) -> list[NewsCandidate]:
	if source == "plx":
		return fetch_plx_page(session, page, timeout)

	page_url = build_page_url(source, page)
	try:
		response = session.get(page_url, timeout=timeout)
	except requests.exceptions.SSLError:
		if source != "bsr":
			raise
		response = session.get(page_url, timeout=timeout, verify=False)
	response.raise_for_status()

	# pecc2.com serves UTF-8 without a charset header, so requests falls back to
	# ISO-8859-1 and mangles the Vietnamese text; decode it as UTF-8 explicitly.
	if source == "pecc2":
		response.encoding = "utf-8"

	base_url = str(SOURCES[source]["base_url"])
	if source == "pvd":
		return parse_pvd_news_items(response.text, base_url)
	if source == "gas":
		return parse_gas_news_items(response.text, base_url)
	if source == "bsr":
		return parse_bsr_news_items(response.text, base_url)
	if source == "nt2":
		return parse_nt2_news_items(response.text, base_url)
	if source == "hdg":
		return parse_hdg_news_items(response.text, base_url)
	if source == "geg":
		return parse_geg_news_items(response.text, base_url)
	if source == "ree":
		return parse_ree_news_items(response.text, base_url)
	if source == "pgv":
		return parse_pgv_news_items(response.text, base_url)
	if source == "pow":
		return parse_pow_news_items(response.text, base_url)
	if source == "pecc2":
		return parse_pecc2_news_items(response.text, base_url)
	return parse_pvs_news_items(response.text, base_url)


def fetch_snippet_from_detail(
	session: requests.Session,
	article_url: str,
	timeout: int = 30,
) -> str:
	if not article_url:
		return ""

	response = session.get(article_url, timeout=timeout)
	response.raise_for_status()
	soup = BeautifulSoup(response.text, "html.parser")

	meta_candidates = [
		soup.select_one("meta[name='description']"),
		soup.select_one("meta[property='og:description']"),
		soup.select_one("meta[itemprop='description']"),
	]
	for node in meta_candidates:
		if node is None:
			continue
		content = normalize_text(str(node.get("content", "")))
		if content:
			return content

	first_para = soup.select_one(".news-detail-content p")
	if first_para is not None:
		text = normalize_text(first_para.get_text(" ", strip=True))
		if text:
			return text

	generic_paragraph_selectors = [
		".news-content p",
		".content-detail p",
		".detail-content p",
		".news-detail p",
		"article p",
	]
	for selector in generic_paragraph_selectors:
		node = soup.select_one(selector)
		if node is None:
			continue
		text = normalize_text(node.get_text(" ", strip=True))
		if text:
			return text

	return ""


def deduplicate(items: Iterable[NewsCandidate]) -> list[NewsCandidate]:
	seen: set[str] = set()
	unique: list[NewsCandidate] = []

	for item in items:
		key = f"{item.date}|{item.title}"
		if key in seen:
			continue
		seen.add(key)
		unique.append(item)

	return unique


def to_output_items(candidates: Iterable[NewsCandidate]) -> list[NewsItem]:
	return [
		NewsItem(
			title=item.title,
			snippet=item.snippet,
			date=item.date,
			url=item.url,
		)
		for item in candidates
	]


def fetch_news(source: str, max_pages: int = 1, delay_seconds: float = 0.3) -> list[NewsItem]:
	all_items: list[NewsCandidate] = []

	with get_session() as session:
		for page in range(1, max_pages + 1):
			page_items = fetch_page(session, source, page)
			if not page_items:
				break
			all_items.extend(page_items)
			if delay_seconds > 0 and page < max_pages:
				time.sleep(delay_seconds)

		unique_items = deduplicate(all_items)

		if source == "ree":
			for item in unique_items:
				try:
					date, snippet = fetch_ree_detail(session, item.url)
					if date:
						item.date = date
					item.snippet = snippet
					time.sleep(0.15)
				except requests.RequestException:
					pass
		elif source in {"pvs", "nt2", "hdg", "plx"}:
			for item in unique_items:
				if item.snippet:
					continue
				try:
					item.snippet = fetch_snippet_from_detail(session, item.url)
					time.sleep(0.15)
				except requests.RequestException:
					item.snippet = ""

	return to_output_items(unique_items)


DEFAULT_OUTPUT = Path("data") / "fetch_press_companies.json"
ALL_SOURCES = list(SOURCES.keys())
DEFAULT_SINCE = "2026-05-01"


def parse_date_tuple(date_str: str) -> tuple[int, int, int] | None:
	match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", date_str or "")
	if match is None:
		return None
	day, month, year = (int(group) for group in match.groups())
	return (year, month, day)


def is_recent(date_str: str, since: tuple[int, int, int]) -> bool:
	"""True if the date is on/after the cutoff. Unknown dates are kept."""
	parsed = parse_date_tuple(date_str)
	if parsed is None:
		return True
	return parsed >= since


def parse_since(value: str) -> tuple[int, int, int]:
	# Accept both YYYY-MM-DD (CLI/ISO) and the dd/mm/yyyy this feed saves as its
	# metadata latest_date, so a stray dd/mm/yyyy doesn't crash the pipeline here.
	value = value.strip()
	if "/" in value:
		day, month, year = (int(part) for part in value.split("/"))
	else:
		year, month, day = (int(part) for part in value.split("-"))
	return (year, month, day)


def latest_date_str(news: list[dict]) -> str | None:
	best: tuple[int, int, int] | None = None
	for item in news:
		parsed = parse_date_tuple(str(item.get("date", "")))
		if parsed is not None and (best is None or parsed > best):
			best = parsed
	if best is None:
		return None
	return f"{best[0]:04d}-{best[1]:02d}-{best[2]:02d}"


def collect_all(sources: list[str], max_pages: int, since: tuple[int, int, int]) -> list[NewsItem]:
	aggregated: list[NewsItem] = []
	for source in sources:
		code = source.upper()
		try:
			items = fetch_news(source=source, max_pages=max_pages)
		except requests.RequestException as exc:
			print(f"  [{code}] error: {exc}")
			continue
		kept = 0
		for item in items:
			if not is_recent(item.date, since):
				continue
			if is_image_url(item.url):
				continue
			item.ticker = [code]
			aggregated.append(item)
			kept += 1
		print(f"  [{code}] {kept} item(s) since cutoff")
	return aggregated


def _news_key(item: dict) -> str:
	return str(item.get("url") or f"{item.get('date', '')}|{item.get('title', '')}")


def merge_news(existing: list[dict], new_items: list[NewsItem]) -> list[dict]:
	"""Merge new items into existing, deduping by url (or date|title) and
	unioning the ticker arrays of duplicates."""
	by_key: dict[str, dict] = {}
	order: list[str] = []
	for record in existing:
		record.setdefault("ticker", [])
		key = _news_key(record)
		if key not in by_key:
			by_key[key] = record
			order.append(key)

	for item in new_items:
		record = asdict(item)
		key = _news_key(record)
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
	output_path.write_text(
		json.dumps(payload, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Fetch IR press news from all company sources into one JSON file, "
		"appending only items newer than the last update."
	)
	parser.add_argument(
		"--sources",
		default="all",
		help="Comma-separated sources to fetch, or 'all' (default). "
		f"Available: {', '.join(ALL_SOURCES)}.",
	)
	parser.add_argument(
		"--pages",
		type=int,
		default=1,
		help="Number of listing pages to fetch per source (default: 1).",
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
	parser.add_argument(
		"--print",
		action="store_true",
		help="Print merged rows to stdout.",
	)
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
	existing_news = existing.get("news", []) if existing else []

	# Incremental: on later runs, only fetch from the latest date already saved.
	saved_latest = existing.get("metadata", {}).get("latest_date") if existing else None
	since = parse_since(saved_latest) if saved_latest else parse_since(args.since)

	print(f"Fetching {len(sources)} source(s) since {since[2]:02d}/{since[1]:02d}/{since[0]} ...")
	new_items = collect_all(sources, max_pages=max(1, args.pages), since=since)
	merged_news = merge_news(existing_news, new_items)

	added = len(merged_news) - len(existing_news)
	payload = {
		"metadata": {
			"last_updated": datetime.now().isoformat(timespec="seconds"),
			"latest_date": latest_date_str(merged_news),
			"sources": sources,
			"total_news": len(merged_news),
		},
		"news": merged_news,
	}
	save_payload(payload, output_path)

	print(f"Saved {len(merged_news)} news rows ({added} new) to {output_path}.")
	if args.print:
		for record in merged_news:
			print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
