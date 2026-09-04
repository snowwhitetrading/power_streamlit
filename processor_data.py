"""
Combined Data Processing Script
Combines functionality from:
- scrape_price.py (price data scraping)
- scrape_volume.py (volume data scraping)
- scrape_type.py (renewable energy type data scraping)
- scrape_water.py (water reservoir data scraping)
- fetch.py (Vinacomin commodity data scraping)
- match.py (data matching and weighted average calculation)
"""

import asyncio
import aiohttp
import csv
import pandas as pd
import re
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl
import certifi
import urllib.parse
import argparse
import time
import requests
import json
import os
import sys
from io import StringIO


HYDROLOGY_SEASON_COLUMNS = [
    "JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ"
]

HYDROLOGY_SEASON_MONTH_MAP = {
    "JJA": 6,
    "JAS": 7,
    "ASO": 8,
    "SON": 9,
    "OND": 10,
    "NDJ": 11,
    "DJF": 12,
    "JFM": 1,
    "FMA": 2,
    "MAM": 3,
    "AMJ": 4,
    "MJJ": 5,
}

HYDROLOGY_FIRST_YEAR_SEASONS = {"JJA", "JAS", "ASO", "SON", "OND", "NDJ", "DJF"}
HYDROLOGY_FORECAST_URL_TEMPLATE = (
    "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/archives/"
    "?type=probabilities&month={month:02d}&year={year}"
)

class DataProcessor:
    def __init__(self):
        # Configuration
        self.default_start_date = datetime(2020, 1, 1)
        self.pmax_default_start_date = datetime(2020, 1, 1)
        # End at the end of *yesterday*: today's data is still incomplete, so we
        # only scrape whole days. Combined with an exclusive append start (day
        # after the last record), each run fetches new whole days and never
        # produces duplicates.
        self.default_end_date = (datetime.now() - timedelta(days=1)).replace(
            hour=23, minute=59, second=59, microsecond=0
        )
        self.csv_encoding = "utf-8-sig"
        
        # Get dashboard directory path
        self.dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Output files (with full paths to dashboard/data/industries_data folder to overwrite existing files)
        self.industries_data_dir = os.path.join(self.dashboard_dir, "data", "industries_data")
        self.price_output = os.path.join(self.industries_data_dir, "fetch_price_smp_monthly.csv")
        self.weighted_output = os.path.join(self.industries_data_dir, "fetch_price_average_monthly.csv")
        self.volume_output = os.path.join(self.industries_data_dir, "fetch_volume_region_monthly.csv")
        self.pmax_output = os.path.join(self.industries_data_dir, "fetch_volume_source_monthly.csv")
        self.water_output = os.path.join(self.industries_data_dir, "fetch_hydro_reservoir_monthly.csv")
        self.vinacomin_output = os.path.join(self.industries_data_dir, "fetch_coal_vinacomin_monthly.csv")
        self.temperature_output = os.path.join(self.industries_data_dir, "fetch_weather_temperature_monthly.csv")
        self.capacity_mobilized_output = os.path.join(self.industries_data_dir, "fetch_capacity_mobilized_monthly.csv")
        self.capacity_installed_output = os.path.join(self.industries_data_dir, "fetch_capacity_installed_monthly.csv")
        self.hydrology_current_output = os.path.join(self.industries_data_dir, "fetch_hydrology_current_monthly.csv")
        self.hydrology_forecast_output = os.path.join(self.industries_data_dir, "fetch_enso_probability_monthly.csv")
        
        # API URLs
        self.price_api_url = "https://www.nsmo.vn/api/services/app/Pages/GetChartGiaBienVM"
        self.temperature_api_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Vietnam cities coordinates for temperature data
        self.vietnam_cities = {
            "Ho_Chi_Minh": {"lat": 10.8231, "lon": 106.6297},
            "Da_Nang": {"lat": 16.0544, "lon": 108.2022},
            "Ha_Noi": {"lat": 21.0285, "lon": 105.8542}
        }
        self.volume_api_url = "https://www.nsmo.vn/api/services/app/Pages/GetChartPhuTaiVM"
        self.pmax_base_url = "https://www.nsmo.vn/HTDThongSoVH"
        self.water_base_url = "https://hochuathuydien.evn.com.vn/PageHoChuaThuyDienEmbedEVN.aspx"
        self.vinacomin_api_url = "https://vinacomin.vn/api/app/commodity/chart"
        self.capacity_mobilized_url = "https://www.nsmo.vn/HTDCongSuatHD"
        self.capacity_installed_url = "https://www.nsmo.vn/TTDCongSuatHD"
        self.hydrology_current_url = "https://ggweather.com/enso/roni.htm"
        self.hydrology_forecast_url_template = HYDROLOGY_FORECAST_URL_TEMPLATE
        
        # Headers
        self.price_header = ["date", "giaBienMB", "giaBienMT", "giaBienMN", "giaBienHT"]
        self.volume_header = ["date", "congSuatMB", "congSuatMT", "congSuatMN", "congSuatHT"]
        self.pmax_header = [
            "date", "max_power_thuong_pham_MW", "max_power_thuong_pham_time", "generation_thuong_pham_mkWh",
            "max_power_dau_cuc_MW", "max_power_dau_cuc_time", "generation_dau_cuc_mkWh",
            "thuy_dien_mkWh", "nhiet_dien_than_mkWh", "tuabin_khi_mkWh", "nhiet_dien_dau_mkWh",
            "dien_gio_mkWh", "dmt_trang_trai_mkWh", "dmt_mai_thuong_pham_mkWh", "dmt_mai_dau_cuc_mkWh",
            "nhap_khau_mkWh", "khac_mkWh"
        ]
        self.water_header = ["date", "region", "reservoir_name", "flood_level", "flood_capacity", "plant_throughput"]
        self.capacity_mobilized_header = [
            "date", "source", "low_season", "high_season"
        ]
        self.capacity_installed_header = [
            "date", "source",
            "installed_total_mw", "low_peak_mw", "high_peak_mw"
        ]
        self.hydrology_current_header = ["date", "oni_value", "enso_type"]
        self.hydrology_forecast_header = ["date", "season", "La Nina prob", "Neutral prob", "El Nino prob"]

    def _repair_mojibake_text(self, value):
        """Repair common UTF-8-vs-CP1252 mojibake without altering normal Vietnamese text."""
        if value is None:
            return None
        text = str(value)
        if not text:
            return text

        # Typical mojibake markers for Vietnamese text.
        if not any(marker in text for marker in ("Ã", "Ä", "Â", "á»", "áº")):
            return text

        try:
            repaired = text.encode("cp1252").decode("utf-8")
            return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text

    def _decode_http_response_text(self, response):
        """Decode HTTP payload defensively to preserve Vietnamese diacritics."""
        raw = response.content
        for encoding in ("utf-8", "utf-8-sig", "cp1258"):
            try:
                decoded = raw.decode(encoding)
                return self._repair_mojibake_text(decoded)
            except UnicodeDecodeError:
                continue
        return self._repair_mojibake_text(raw.decode("latin1", errors="replace"))

    # Price Data Scraping
    async def fetch_price_data(self, session, date):
        date_str = date.strftime("%d/%m/%Y")
        print(f"Fetching price data for {date_str}...")
        
        try:
            async with session.get(self.price_api_url, params={"day": date_str}, ssl=False) as response:
                if response.status != 200:
                    print(f"Failed to fetch price data for {date_str}: HTTP {response.status}")
                    return None
                data = await response.json()
                
                if (data.get("result") and 
                    data["result"].get("status") and 
                    data["result"].get("data")):
                    return {
                        "date_str": date_str,
                        "thoiGianCapNhat": data["result"]["data"].get("thoiGianCapNhat"),
                        "giaBiens": data["result"]["data"].get("giaBiens", [])
                    }
                else:
                    print(f"No price data for {date_str}")
                    return None
        except Exception as e:
            print(f"Error fetching price data for {date_str}: {str(e)}")
            return None

    # Volume Data Scraping
    async def fetch_volume_data(self, session, date):
        date_str = date.strftime("%d/%m/%Y")
        print(f"Fetching volume data for {date_str}...")
        
        try:
            async with session.get(self.volume_api_url, params={"day": date_str}, ssl=False) as response:
                if response.status != 200:
                    print(f"Failed to fetch volume data for {date_str}: HTTP {response.status}")
                    return None
                data = await response.json()
                
                if (data.get("result") and 
                    data["result"].get("status") and 
                    data["result"].get("data")):
                    return {
                        "date_str": date_str,
                        "thoiGianCapNhat": data["result"]["data"].get("thoiGianCapNhat"),
                        "phuTais": data["result"]["data"].get("phuTais", [])
                    }
                else:
                    print(f"No volume data for {date_str}")
                    return None
        except Exception as e:
            print(f"Error fetching volume data for {date_str}: {str(e)}")
            return None

    # Renewable Energy Data Scraping
    async def fetch_renewable_html(self, session, date, xsrf_token=None):
        date_str = date.strftime("%d/%m/%Y")
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36',
                'Accept': 'text/plain, */*; q=0.01',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br, zstd',
                'Connection': 'keep-alive',
                'Content-Type': 'application/text; charset=utf-8',
                'X-Requested-With': 'XMLHttpRequest',
                'Referer': 'https://www.nsmo.vn/HeThongDien',
                'sec-ch-ua': '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"',
                'sec-fetch-dest': 'empty',
                'sec-fetch-mode': 'cors',
                'sec-fetch-site': 'same-origin',
            }
            if xsrf_token:
                headers['x-xsrf-token'] = xsrf_token
            
            async with session.get(self.pmax_base_url, params={"day": date_str}, headers=headers, ssl=False, timeout=timeout) as resp:
                if resp.status != 200:
                    print(f"Failed to fetch pmax data for {date_str}: HTTP {resp.status}")
                    return None
                return await resp.text()
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            print(f"Network error fetching pmax data for {date_str}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching pmax data for {date_str}: {e}")
            return None

    def parse_renewable_html(self, html, date):
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            
            import re
            
            # Enhanced regex patterns based on the actual HTML structure
            patterns = {
                # Power capacity patterns (MW)
                'max_power_thuong_pham_MW': r'Công suất lớn nhất trong ngày.*?(\d+(?:[,\.]\d+)?)\s*MW\s*\(Lúc\s+(\d{1,2}:\d{2})\)',
                'max_power_dau_cuc_MW': r'Công suất lớn nhất trong ngày.*?(\d+(?:[,\.]\d+)?)\s*MW\s*\(Lúc\s+(\d{1,2}:\d{2})\).*?Công suất lớn nhất trong ngày.*?(\d+(?:[,\.]\d+)?)\s*MW\s*\(Lúc\s+(\d{1,2}:\d{2})\)',
                
                # Generation patterns (million kWh)
                'generation_thuong_pham_mkWh': r'Sản lượng điện sản xuất và nhập khẩu.*?(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'generation_dau_cuc_mkWh': r'Sản lượng điện sản xuất và nhập khẩu.*?(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh.*?Sản lượng điện sản xuất và nhập khẩu.*?(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                
                # Energy source breakdown patterns
                'thuy_dien_mkWh': r'Thủy điện\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'nhiet_dien_than_mkWh': r'Nhiệt điện than\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'tuabin_khi_mkWh': r'Tuabin khí.*?(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'nhiet_dien_dau_mkWh': r'Nhiệt điện dầu\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'dien_gio_mkWh': r'Điện gió\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'dmt_trang_trai_mkWh': r'ĐMT trang trại\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'dmt_mai_thuong_pham_mkWh': r'ĐMT mái nhà \(ước tính thương phẩm\)\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'dmt_mai_dau_cuc_mkWh': r'ĐMT mái nhà \(ước tính đầu cực\)\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'nhap_khau_mkWh': r'Nhập khẩu điện\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh',
                'khac_mkWh': r'Khác \(Sinh khối, Diesel Nam.*?\)\s+(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh'
            }
            
            result = [date.strftime("%Y-%m-%d")]
            
            # Extract max power for thuong pham
            thuong_pham_match = re.search(patterns['max_power_thuong_pham_MW'], text, re.IGNORECASE | re.DOTALL)
            if thuong_pham_match:
                power_value = thuong_pham_match.group(1).replace(',', '.')
                time_value = thuong_pham_match.group(2)
                result.extend([power_value, time_value])
            else:
                result.extend(["0", "00:00"])
            
            # Extract generation for thuong pham
            gen_thuong_match = re.search(patterns['generation_thuong_pham_mkWh'], text, re.IGNORECASE)
            result.append(gen_thuong_match.group(1).replace(',', '.') if gen_thuong_match else "0")
            
            # Extract max power for dau cuc (look for second occurrence)
            dau_cuc_power_pattern = r'Tính với số liệu ĐMT mái nhà \(ước tính đầu cực\).*?Công suất lớn nhất trong ngày.*?(\d+(?:[,\.]\d+)?)\s*MW\s*\(Lúc\s+(\d{1,2}:\d{2})\)'
            dau_cuc_match = re.search(dau_cuc_power_pattern, text, re.IGNORECASE | re.DOTALL)
            if dau_cuc_match:
                power_value = dau_cuc_match.group(1).replace(',', '.')
                time_value = dau_cuc_match.group(2)
                result.extend([power_value, time_value])
            else:
                result.extend(["0", "00:00"])
            
            # Extract generation for dau cuc (look for second occurrence)
            gen_dau_cuc_pattern = r'Tính với số liệu ĐMT mái nhà \(ước tính đầu cực\).*?Sản lượng điện sản xuất và nhập khẩu.*?(\d+(?:[,\.]\d+)?)\s*triệu\s*kWh'
            gen_dau_cuc_match = re.search(gen_dau_cuc_pattern, text, re.IGNORECASE | re.DOTALL)
            result.append(gen_dau_cuc_match.group(1).replace(',', '.') if gen_dau_cuc_match else "0")
            
            # Extract energy source breakdown
            energy_sources = [
                'thuy_dien_mkWh', 'nhiet_dien_than_mkWh', 'tuabin_khi_mkWh', 'nhiet_dien_dau_mkWh',
                'dien_gio_mkWh', 'dmt_trang_trai_mkWh', 'dmt_mai_thuong_pham_mkWh', 'dmt_mai_dau_cuc_mkWh',
                'nhap_khau_mkWh', 'khac_mkWh'
            ]
            
            for source in energy_sources:
                if source in patterns:
                    match = re.search(patterns[source], text, re.IGNORECASE)
                    value = match.group(1).replace(',', '.') if match else "0"
                    result.append(value)
                else:
                    result.append("0")
            
            return result
            
        except Exception as e:
            print(f"Parsing error for renewable data on {date}: {e}")
            return None

    # Water Reservoir Data Scraping
    async def fetch_water_data(self, session, date, max_retries=3):
        date_str = date.strftime("%d/%m/%Y %H:%M")  # required by the EVN API query
        stored_date = date.strftime("%Y-%m-%d %H:%M:%S")  # unified CSV format
        encoded_date = urllib.parse.quote(date_str)
        url = f"{self.water_base_url}?td={encoded_date}&vm=&lv=&hc="
        
        print(f"Fetching water reservoir data for {date_str}...")
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                }
                
                async with session.get(url, headers=headers, ssl=False, timeout=30) as response:
                    if response.status != 200:
                        print(f"Failed to fetch water data for {date_str}: HTTP {response.status}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return []
                    
                    html = await response.text(encoding='utf-8')
                    html = self._repair_mojibake_text(html)
                    soup = BeautifulSoup(html, 'html.parser')
                    rows = []
                    current_region = None
                    
                    # Debug: Print total number of tables and rows found
                    tables = soup.find_all('table')
                    print(f"Found {len(tables)} tables")
                    
                    # Look for the main data table
                    for table in tables:
                        table_rows = table.find_all('tr')
                        print(f"Table has {len(table_rows)} rows")
                        
                        for tr in table_rows:
                            tr_classes = tr.get('class', [])
                            
                            # Enhanced region header detection - try multiple methods
                            region_found = False
                            
                            # Method 1: Check for specific CSS classes
                            if ('tralter' in tr_classes or 
                                any('header' in cls.lower() for cls in tr_classes) or
                                any('region' in cls.lower() for cls in tr_classes)):
                                region_found = True
                            
                            # Method 2: Check if row contains region-like text patterns
                            if not region_found:
                                row_text = tr.get_text(strip=True)
                                region_patterns = ['Tây Nguyên', 'Tây Bắc Bộ', 'Đông Bắc Bộ', 'Bắc Trung Bộ', 'Nam Trung Bộ', 'Đông Nam Bộ']
                                for pattern in region_patterns:
                                    if pattern in row_text and len(tr.find_all('td')) <= 3:  # Region headers usually have few columns
                                        region_found = True
                                        break
                            
                            # Method 3: Check for specific styling or attributes
                            if not region_found:
                                # Look for bold text or specific background colors
                                if tr.find('strong') or tr.find('b'):
                                    cell_text = tr.get_text(strip=True)
                                    if any(pattern in cell_text for pattern in ['Tây Nguyên', 'Tây Bắc Bộ', 'Đông Bắc Bộ', 'Bắc Trung Bộ', 'Nam Trung Bộ', 'Đông Nam Bộ']):
                                        region_found = True
                            
                            if region_found:
                                td = tr.find('td')
                                if td:
                                    # Try multiple ways to extract region name
                                    region_text = None
                                    
                                    # Try strong tag first
                                    strong_tag = td.find('strong')
                                    if strong_tag:
                                        region_text = strong_tag.get_text(strip=True)
                                    else:
                                        # Try bold tag
                                        b_tag = td.find('b')
                                        if b_tag:
                                            region_text = b_tag.get_text(strip=True)
                                        else:
                                            # Use full td text
                                            region_text = td.get_text(strip=True)
                                    
                                    if region_text:
                                        region_text = self._repair_mojibake_text(region_text)
                                        # Enhanced region mapping with fuzzy matching
                                        region_mapping = {
                                            'Tây Nguyên': 'Tây Nguyên',
                                            'Tây Bắc Bộ': 'Tây Bắc Bộ', 
                                            'Đông Bắc Bộ': 'Đông Bắc Bộ',
                                            'Bắc Trung Bộ': 'Bắc Trung Bộ',
                                            'Nam Trung Bộ': 'Nam Trung Bộ',
                                            'Đông Nam Bộ': 'Đông Nam Bộ'
                                        }
                                        
                                        # Check if the text contains any known region
                                        for region_key in region_mapping.keys():
                                            if region_key in region_text:
                                                current_region = region_mapping[region_key]
                                                print(f"Found region: {current_region}")
                                                break
                                        continue
                            
                            # Check if this is a data row (has multiple td elements)
                            tds = tr.find_all('td')
                            if len(tds) >= 8 and current_region:  # Need at least 8 columns for our data
                                try:
                                    # Extract reservoir name (first column)
                                    reservoir_cell = tds[0].get_text(strip=True)
                                    if not reservoir_cell or reservoir_cell in ['Tên hồ', 'Thời điểm', '']:
                                        continue  # Skip header rows
                                    
                                    # Clean up reservoir name by removing the timestamp part
                                    reservoir_name = reservoir_cell.split('Đồng bộ lúc:')[0].strip()
                                    reservoir_name = self._repair_mojibake_text(reservoir_name)
                                    if not reservoir_name:
                                        continue
                                    
                                    # Validate that this is actually a reservoir name (not a region or header)
                                    if any(region in reservoir_name for region in ['Tây Nguyên', 'Tây Bắc Bộ', 'Đông Bắc Bộ', 'Bắc Trung Bộ', 'Nam Trung Bộ', 'Đông Nam Bộ']):
                                        continue
                                    
                                    # Extract water level (column index 2 - Htl)
                                    flood_level_text = tds[2].get_text(strip=True) if len(tds) > 2 else ""
                                    
                                    # Extract flood capacity (column index 5 - Qvề)
                                    flood_capacity_text = tds[5].get_text(strip=True) if len(tds) > 5 else ""
                                    
                                    # Extract plant throughput (column index 8 - Qxm)
                                    plant_throughput_text = tds[8].get_text(strip=True) if len(tds) > 8 else ""
                                    
                                    # Clean numeric values
                                    def clean_numeric(value):
                                        if not value or value == '-' or value == '':
                                            return ""
                                        # Remove any non-numeric characters except decimal points and commas
                                        cleaned = re.sub(r'[^\d.,\-]', '', value)
                                        # Replace comma with dot for decimal
                                        cleaned = cleaned.replace(',', '.')
                                        try:
                                            float(cleaned)
                                            return cleaned
                                        except:
                                            return ""
                                    
                                    flood_level = clean_numeric(flood_level_text)
                                    flood_capacity = clean_numeric(flood_capacity_text)
                                    plant_throughput = clean_numeric(plant_throughput_text)
                                    
                                    # Only add row if we have valid data
                                    if reservoir_name and (flood_level or flood_capacity or plant_throughput):
                                        row = [
                                            stored_date,
                                            current_region,
                                            reservoir_name,
                                            flood_level,
                                            flood_capacity,
                                            plant_throughput
                                        ]
                                        rows.append(row)
                                        if current_region in ['Tây Nguyên', 'Tây Bắc Bộ']:
                                            print(f"Added: {current_region} - {reservoir_name}")
                                
                                except (IndexError, AttributeError, ValueError) as e:
                                    print(f"Error processing row: {e}")
                                    continue
                    
                    print(f"Successfully extracted {len(rows)} water reservoir records for {date_str}")
                    return rows
                    
            except Exception as e:
                print(f"Error fetching water data for {date_str} (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return []
        
        return []

    def scrape_vinacomin_data(self):
        """Scrape Vinacomin commodity data from API endpoint with enhanced functionality"""
        print("🏭 Scraping Vinacomin commodity data...")
        
        # Enhanced headers matching the latest API requirements
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7,und;q=0.6',
            'connection': 'keep-alive',
            'host': 'vinacomin.vn',
            'language': 'vi',
            'referer': 'https://vinacomin.vn/vi/rate',
            'sec-ch-ua': '"Not;A=Brand";v="99", "Google Chrome";v="139", "Chromium";v="139"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36'
        }
        
        try:
            response = requests.get(self.vinacomin_api_url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            data = response.json()
            
            if data.get('code') == 0 and 'data' in data:
                processed_records = []
                
                for commodity in data['data']:
                    commodity_name = commodity.get('label', 'Unknown')
                    
                    # Process each price record for this commodity
                    for price_data in commodity.get('prices', []):
                        record = {
                            'commodity': commodity_name,
                            'update_date': price_data.get('updateDate'),
                            'week': price_data.get('week'),
                            'price': price_data.get('price'),
                            'currency_per_unit': price_data.get('currencyPerUnit'),
                            'percent_change': price_data.get('percentChange')
                        }
                        processed_records.append(record)
                
                # Create DataFrame
                df = pd.DataFrame(processed_records)
                
                # Convert data types and process
                if not df.empty:
                    df['date'] = pd.to_datetime(df['update_date'])
                    df = df.drop(columns=['update_date'])
                    df['week'] = df['week'].astype(int)
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                    df['percent_change'] = pd.to_numeric(df['percent_change'], errors='coerce')
                    
                    # Sort by commodity and week
                    df = df.sort_values(['commodity', 'week']).reset_index(drop=True)
                    
                    # Save to CSV
                    df.to_csv(self.vinacomin_output, index=False, encoding=self.csv_encoding)
                    
                    print(f"✅ Vinacomin data saved to {self.vinacomin_output}")
                    print(f"📊 Total records: {len(processed_records)}")
                    print(f"🏷️ Commodities found: {df['commodity'].nunique()}")
                    print("\n📦 Commodity details:")
                    
                    for commodity in df['commodity'].unique():
                        commodity_data = df[df['commodity'] == commodity]
                        latest_price = commodity_data['price'].iloc[-1] if not commodity_data.empty else 0
                        avg_change = commodity_data['percent_change'].mean()
                        count = len(commodity_data)
                        currency = commodity_data['currency_per_unit'].iloc[0] if not commodity_data.empty else 'N/A'
                        
                        print(f"   - {commodity}: {count} records")
                        print(f"     Latest: {latest_price:,.1f} {currency}")
                        print(f"     Avg change: {avg_change:.2f}%")
                else:
                    print("⚠️ No data processed - DataFrame is empty")
                    
            else:
                print(f"❌ API returned error: {data.get('msg', 'Unknown error')}")
                print(f"Response code: {data.get('code', 'Unknown')}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making API request: {e}")
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON response: {e}")
        except Exception as e:
            print(f"❌ Unexpected error in Vinacomin scraping: {e}")

    # Temperature Data Scraping (Open-Meteo API - FREE, no API key required)
    def scrape_temperature_data(self, start_date=None, end_date=None):
        """
        Scrape historical daily mean temperature for Vietnamese cities using Open-Meteo Archive API
        
        Data source: Open-Meteo (https://open-meteo.com/)
        - FREE API, no API key required
        - Historical data available from 1940 to present
        - Cities: Ho Chi Minh, Da Nang, Ha Noi
        """
        print("🌡️ Scraping temperature data for Vietnamese cities...")

        start_date = start_date or self.default_start_date
        end_date_dt = end_date or datetime.now()

        # Append-aware: only fetch from the day after the last recorded date so we
        # extend the existing history instead of re-downloading everything.
        append_start, _ = self._detect_append_start(self.temperature_output, "date")
        if append_start is not None:
            start_date = max(start_date, append_start)

        if start_date > end_date_dt:
            print(f"✅ Temperature data already up to date (last date: {(append_start - timedelta(days=1)).strftime('%Y-%m-%d')})")
            return None

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date_dt.strftime("%Y-%m-%d")

        print(f"📅 Date range: {start_date_str} to {end_date_str}")
        
        all_data = None
        
        for city_name, coords in self.vietnam_cities.items():
            try:
                params = {
                    "latitude": coords["lat"],
                    "longitude": coords["lon"],
                    "start_date": start_date_str,
                    "end_date": end_date_str,
                    "daily": "temperature_2m_mean",
                    "timezone": "Asia/Ho_Chi_Minh"
                }
                
                print(f"  Fetching data for {city_name}...")
                response = requests.get(self.temperature_api_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    daily = data["daily"]
                    
                    df = pd.DataFrame({
                        "date": daily["time"],
                        f"{city_name}_temp_mean": daily["temperature_2m_mean"]
                    })
                    df["date"] = pd.to_datetime(df["date"])
                    print(f"  ✅ Got {len(df)} days of data for {city_name}")
                    
                    if all_data is None:
                        all_data = df
                    else:
                        all_data = all_data.merge(df, on="date", how="outer")
                else:
                    print(f"  ❌ Error for {city_name}: HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error fetching {city_name}: {e}")
        
        if all_data is not None:
            # Sort by date
            all_data = all_data.sort_values("date").reset_index(drop=True)

            # Merge with existing history so updates append new points instead of replacing.
            if os.path.exists(self.temperature_output):
                existing_data = pd.read_csv(self.temperature_output, encoding=self.csv_encoding)
                if not existing_data.empty and "date" in existing_data.columns:
                    existing_data["date"] = pd.to_datetime(existing_data["date"], errors="coerce")
                    all_data = pd.concat([existing_data, all_data], ignore_index=True)
                    all_data = all_data.drop_duplicates(subset=["date"], keep="last")
                    all_data = all_data.sort_values("date").reset_index(drop=True)

            # Save to CSV
            all_data.to_csv(self.temperature_output, index=False, encoding=self.csv_encoding)
            
            print(f"\n✅ Temperature data saved to {self.temperature_output}")
            print(f"   Total records: {len(all_data)} days")
            print(f"   Date range: {all_data['date'].min().strftime('%Y-%m-%d')} to {all_data['date'].max().strftime('%Y-%m-%d')}")
            
            # Show summary statistics
            print("\n📊 Temperature Statistics (°C):")
            print("-" * 70)
            print(f"{'City':<15} {'Mean':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
            print("-" * 70)
            
            for city in self.vietnam_cities.keys():
                col = f"{city}_temp_mean"
                if col in all_data.columns:
                    mean_temp = all_data[col].mean()
                    min_temp = all_data[col].min()
                    max_temp = all_data[col].max()
                    std_temp = all_data[col].std()
                    print(f"{city:<15} {mean_temp:>10.1f} {min_temp:>10.1f} {max_temp:>10.1f} {std_temp:>10.1f}")
            
            print("-" * 70)
            
            return all_data
        else:
            print("❌ Failed to fetch temperature data")
            return None

    # Data Processing and Matching
    def _classify_hydrology_enso_type(self, oni_value):
        if oni_value < -1.5:
            return "strong la nina"
        if -1.5 < oni_value < -1:
            return "medium la nina"
        if -1 < oni_value < -0.5:
            return "weak la nina"
        if oni_value > 1.5:
            return "strong el nino"
        if 1 < oni_value < 1.5:
            return "medium el nino"
        if 0.5 < oni_value < 1:
            return "weak el nino"
        return "neutral"

    def _parse_hydrology_float(self, value):
        if value is None:
            return None
        cleaned = str(value).replace("\xa0", " ").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _hydrology_season_cell_to_date(self, season_range, season_col):
        left_year_str, right_year_str = str(season_range).split("-", maxsplit=1)
        left_year = int(left_year_str)
        right_year = int(right_year_str)

        month = HYDROLOGY_SEASON_MONTH_MAP[season_col]
        year = left_year if season_col in HYDROLOGY_FIRST_YEAR_SEASONS else right_year
        return datetime(year, month, 1)

    def _find_hydrology_table(self, soup):
        required_headers = {"Season", *HYDROLOGY_SEASON_COLUMNS}
        for table in soup.find_all("table"):
            header_row = table.find("tr")
            if not header_row:
                continue
            headers = [cell.get_text(" ", strip=True) for cell in header_row.find_all(["th", "td"])]
            if required_headers.issubset(set(headers)):
                return table
        raise ValueError("Cannot find hydrology current table with expected headers")

    def scrape_hydrology_current_monthly_data(self, start_date=None, end_date=None, source_url=None):
        """Fetch monthly hydrology current index (RONI table) and save as CSV."""
        print("[Hydrology] Scraping hydrology current monthly data...")

        start_date = start_date or datetime(2010, 1, 1)
        end_date = end_date or self.default_end_date
        source_url = source_url or self.hydrology_current_url

        response = requests.get(source_url, timeout=60)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        table = self._find_hydrology_table(soup)

        records = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 14:
                continue

            season_range = tds[1].get_text(" ", strip=True)
            if "-" not in season_range:
                continue

            for idx, season_col in enumerate(HYDROLOGY_SEASON_COLUMNS, start=2):
                oni_value = self._parse_hydrology_float(tds[idx].get_text(" ", strip=True))
                if oni_value is None:
                    continue

                record_date = self._hydrology_season_cell_to_date(season_range, season_col)
                if record_date < start_date or record_date > end_date:
                    continue

                records.append(
                    {
                        "date": record_date.date(),
                        "oni_value": oni_value,
                        "enso_type": self._classify_hydrology_enso_type(oni_value),
                    }
                )

        if not records:
            raise ValueError("No hydrology current records parsed from source")

        df = pd.DataFrame(records)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        df = df[self.hydrology_current_header]
        df.to_csv(self.hydrology_current_output, index=False, encoding=self.csv_encoding)

        print(f"[Hydrology] Data saved to {self.hydrology_current_output}")
        print(f"[Hydrology] Total records: {len(df)}")
        if not df.empty:
            print(f"[Hydrology] Date range: {df['date'].min()} to {df['date'].max()}")
        return df

    def _iter_month_start_dates(self, start_date, end_date):
        current = datetime(start_date.year, start_date.month, 1)
        while current <= end_date:
            yield current
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)

    def _extract_hydrology_forecast_table(self, html_text):
        tables = pd.read_html(StringIO(html_text))
        for table in tables:
            cols = [str(c).strip() for c in table.columns]
            if "Season" in cols and any("Neutral" in c for c in cols) and any("El" in c for c in cols):
                return table.copy()
        raise ValueError("Could not find NOAA hydrology forecast probability table")

    def _normalize_hydrology_forecast_table(self, table, issue_month):
        col_map = {}
        for col in table.columns:
            col_txt = str(col).strip().lower()
            if col_txt == "season":
                col_map[col] = "season"
            elif "la" in col_txt and ("nina" in col_txt or "niña" in col_txt or "ni" in col_txt):
                col_map[col] = "La Nina prob"
            elif "neutral" in col_txt:
                col_map[col] = "Neutral prob"
            elif "el" in col_txt and ("nino" in col_txt or "niño" in col_txt or "ni" in col_txt):
                col_map[col] = "El Nino prob"

        required = {"season", "La Nina prob", "Neutral prob", "El Nino prob"}
        if set(col_map.values()) != required:
            raise ValueError(f"Unexpected NOAA forecast columns: {list(table.columns)}")

        df = table.rename(columns=col_map)[["season", "La Nina prob", "Neutral prob", "El Nino prob"]].copy()

        # Keep only season code from values like "AMJ Apr May Jun".
        df["season"] = (
            df["season"].astype(str).str.strip().str.split().str[0].str.upper()
        )

        season_pattern = re.compile(r"^[A-Z]{3}$")
        df = df[df["season"].apply(lambda x: bool(season_pattern.match(str(x))))].copy()

        for col in ["La Nina prob", "Neutral prob", "El Nino prob"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["La Nina prob", "Neutral prob", "El Nino prob"])
        df["date"] = issue_month.date()
        return df[self.hydrology_forecast_header].reset_index(drop=True)

    def scrape_hydrology_forecast_monthly_data(self, start_date=None, end_date=None, source_url_template=None):
        """Fetch NOAA hydrology forecast probabilities by issue month and save as CSV."""
        print("[HydrologyForecast] Scraping monthly forecast probability data...")

        start_date = start_date or datetime(2026, 4, 1)
        end_date = end_date or self.default_end_date
        source_url_template = source_url_template or self.hydrology_forecast_url_template

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            }
        )

        rows = []
        for issue_month in self._iter_month_start_dates(start_date, end_date):
            url = source_url_template.format(month=issue_month.month, year=issue_month.year)
            try:
                response = session.get(url, timeout=45)
                if response.status_code != 200:
                    print(f"[HydrologyForecast] Skip {issue_month.strftime('%Y-%m')}: HTTP {response.status_code}")
                    continue

                table = self._extract_hydrology_forecast_table(response.text)
                month_df = self._normalize_hydrology_forecast_table(table, issue_month)
                if month_df.empty:
                    print(f"[HydrologyForecast] Skip {issue_month.strftime('%Y-%m')}: empty table")
                    continue

                rows.append(month_df)
                print(f"[HydrologyForecast] OK {issue_month.strftime('%Y-%m')}: {len(month_df)} seasons")
            except Exception as exc:
                print(f"[HydrologyForecast] Skip {issue_month.strftime('%Y-%m')}: {exc}")

        if not rows:
            df = pd.DataFrame(columns=self.hydrology_forecast_header)
        else:
            df = pd.concat(rows, ignore_index=True)

        df.to_csv(self.hydrology_forecast_output, index=False, encoding=self.csv_encoding)

        print(f"[HydrologyForecast] Data saved to {self.hydrology_forecast_output}")
        print(f"[HydrologyForecast] Total records: {len(df)}")
        if not df.empty:
            print(f"[HydrologyForecast] Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    # Data Processing and Matching
    def process_weighted_averages(self):
        """Process volume and price data to calculate weighted averages"""
        try:
            print("Processing weighted average prices...")
            
            # Read the data files
            volume_df = pd.read_csv(self.volume_output, encoding=self.csv_encoding)
            price_df = pd.read_csv(self.price_output, encoding=self.csv_encoding)
            
            # Convert datetime columns. The source CSVs mix date formats
            # (M/D/YYYY, D/M/YYYY, and ISO), so a bare to_datetime locks onto the
            # first row's format and raises on the first row that differs (e.g.
            # "13/1/2020 0:30"). format="mixed" parses each value on its own and
            # coerce drops any unparseable row instead of aborting the whole step.
            volume_df['date'] = pd.to_datetime(volume_df['date'], format="mixed", errors="coerce")
            price_df['date'] = pd.to_datetime(price_df['date'], format="mixed", errors="coerce")
            volume_df = volume_df.dropna(subset=['date'])
            price_df = price_df.dropna(subset=['date'])
            
            # Remove MB, MT, MN columns from volume data
            volume_df = volume_df[['date', 'congSuatHT']]
            
            # Merge the dataframes on datetime
            merged_df = pd.merge(volume_df, 
                               price_df[['date', 'giaBienHT']], 
                               on='date', 
                               how='inner')
            
            # Remove rows where price < 50
            merged_df = merged_df[merged_df['giaBienHT'] >= 50]
            
            # Add date column for grouping
            merged_df['date'] = merged_df['date'].dt.date
            
            # Calculate volume-weighted average price for each day
            result_df = (merged_df.groupby('date')
                        .apply(lambda x: np.average(x['giaBienHT'], 
                                                  weights=x['congSuatHT']))
                        .reset_index())
            result_df.columns = ['date', 'weighted_avg_price']
            
            # Add volume sum for reference
            volume_sums = (merged_df.groupby('date')['congSuatHT']
                          .sum()
                          .reset_index())
            result_df = result_df.merge(volume_sums, on='date')
            
            # Sort by date
            result_df = result_df.sort_values('date')
            
            # Save the results
            result_df.to_csv(self.weighted_output, index=False, encoding=self.csv_encoding)
            print(f"Weighted average prices saved to {self.weighted_output}")
            
            return result_df
            
        except Exception as e:
            print(f"Error processing weighted averages: {str(e)}")
            return None

    async def scrape_price_data(self, start_date=None, end_date=None):
        """Scrape price data – appends from the last existing date."""
        end_date = end_date or self.default_end_date

        # Detect last existing date and adjust start
        append_start, file_exists = self._detect_append_start(self.price_output, "date")
        if append_start is not None:
            effective_start = max(start_date, append_start) if start_date else append_start
        else:
            effective_start = start_date or self.default_start_date

        if effective_start > end_date:
            print(f"✅ Price data already up to date (last date: {(append_start - timedelta(days=1)).strftime('%Y-%m-%d')})")
            return

        print(f"📅 Fetching price data from {effective_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        start_date = effective_start
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        semaphore = asyncio.Semaphore(50)

        async def fetch_with_semaphore(session, date):
            async with semaphore:
                return await self.fetch_price_data(session, date)

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_with_semaphore(session, date) for date in dates]
            results = await asyncio.gather(*tasks)
            
            write_mode = "a" if file_exists else "w"
            with open(self.price_output, mode=write_mode, newline="", encoding=self.csv_encoding) as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.price_header)
                
                for result in results:
                    if result:
                        for item in result["giaBiens"]:
                            writer.writerow([
                                (item.get("thoiGian") or "").replace("T", " "),
                                item.get("giaBienMB"),
                                item.get("giaBienMT"),
                                item.get("giaBienMN"),
                                item.get("giaBienHT"),
                            ])

        self._dedupe_csv(self.price_output, ["date"])
        print(f"✅ Price data {'appended to' if file_exists else 'saved to'} {self.price_output}")

    async def scrape_volume_data(self, start_date=None, end_date=None):
        """Scrape volume data – appends from the last existing date."""
        end_date = end_date or self.default_end_date

        # Detect last existing date and adjust start
        append_start, file_exists = self._detect_append_start(self.volume_output, "date")
        if append_start is not None:
            effective_start = max(start_date, append_start) if start_date else append_start
        else:
            effective_start = start_date or self.default_start_date

        if effective_start > end_date:
            print(f"✅ Volume data already up to date (last date: {(append_start - timedelta(days=1)).strftime('%Y-%m-%d')})")
            return

        print(f"📅 Fetching volume data from {effective_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        start_date = effective_start
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        semaphore = asyncio.Semaphore(50)

        async def fetch_with_semaphore(session, date):
            async with semaphore:
                return await self.fetch_volume_data(session, date)

        async with aiohttp.ClientSession() as session:
            tasks = [fetch_with_semaphore(session, date) for date in dates]
            results = await asyncio.gather(*tasks)
            
            write_mode = "a" if file_exists else "w"
            with open(self.volume_output, mode=write_mode, newline="", encoding=self.csv_encoding) as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.volume_header)
                
                for result in results:
                    if result:
                        for item in result["phuTais"]:
                            writer.writerow([
                                (item.get("thoiGian") or "").replace("T", " "),
                                item.get("congSuatMB"),
                                item.get("congSuatMT"),
                                item.get("congSuatMN"),
                                item.get("congSuatHT"),
                            ])

        self._dedupe_csv(self.volume_output, ["date"])
        print(f"✅ Volume data {'appended to' if file_exists else 'saved to'} {self.volume_output}")

    def scrape_power_capacity_monthly_data(self, start_date=None, end_date=None, monthly=False, rebuild=False, max_workers=20):
        """Scrape P Max (Maximum Power) data – appends from the last existing date."""
        end_date = end_date or self.default_end_date

        # Detect last existing date and adjust start (or rebuild full history)
        if rebuild:
            append_start = None
            file_exists = False
            effective_start = start_date or self.pmax_default_start_date
        else:
            append_start, file_exists = self._detect_append_start(self.pmax_output, "date")
            if append_start is not None:
                effective_start = max(start_date, append_start) if start_date else append_start
            else:
                effective_start = start_date or self.pmax_default_start_date

        if effective_start > end_date:
            print(f"[PMax] Data already up to date (last date: {(append_start - timedelta(days=1)).strftime('%Y-%m-%d')})")
            return

        print(f"[PMax] Fetching data from {effective_start.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        start_date = effective_start
        
        if monthly:
            dates = list(self._iter_monthly_dates(start_date, end_date))
            print(f"[PMax] Monthly mode: {len(dates)} months to fetch")
        else:
            dates = list(self._iter_daily_dates(start_date, end_date))
            print(f"[PMax] Daily mode: {len(dates)} days to fetch")

        # Get initial session with XSRF token
        parent_url = "https://www.nsmo.vn/HeThongDien"
        init_session = self._create_capacity_session(parent_url)
        cookies_snapshot = {c.name: c.value for c in init_session.cookies}
        xsrf_token = init_session.headers.get("x-xsrf-token", "")

        def fetch_day(day):
            try:
                s = requests.Session()
                s.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                    "Accept": "text/plain, */*; q=0.01",
                    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Accept-Encoding": "gzip, deflate, br, zstd",
                    "Content-Type": "application/text; charset=utf-8",
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": parent_url,
                })
                for name, value in cookies_snapshot.items():
                    s.cookies.set(name, value)
                if xsrf_token:
                    s.headers["x-xsrf-token"] = xsrf_token
                
                # Fetch page
                day_str = day.strftime("%d/%m/%Y")
                resp = s.get(self.pmax_base_url, params={"day": day_str}, timeout=30, verify=False)
                resp.raise_for_status()
                
                html_text = self._decode_http_response_text(resp)
                if html_text and len(html_text) > 100:  # Check if we got real data
                    result = self.parse_renewable_html(html_text, day)
                    if result:
                        print(f"[OK] {day.strftime('%Y-%m-%d')}: {result[0]}")
                        return result
                else:
                    print(f"[WARN] {day.strftime('%Y-%m-%d')}: empty response")
                return None
            except Exception as e:
                print(f"[ERR] {day.strftime('%Y-%m-%d')}: {str(e)[:50]}")
                return None

        all_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in executor.map(fetch_day, dates):
                if result:
                    all_rows.append(result)

        # Sort rows by date
        all_rows.sort(key=lambda x: x[0])

        write_mode = "a" if file_exists else "w"
        with open(self.pmax_output, write_mode, newline="", encoding=self.csv_encoding) as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.pmax_header)
            writer.writerows(all_rows)

        self._dedupe_csv(self.pmax_output, ["date"])
        print(f"[PMax] Data {'appended to' if file_exists else 'saved to'} {self.pmax_output}")
        print(f"[PMax] Total records saved: {len(all_rows)}")
        if all_rows:
            print(f"[PMax] Date range: {all_rows[0][0]} to {all_rows[-1][0]}")

    async def scrape_water_data(self, start_date=None, end_date=None, rebuild=False):
        """Scrape water reservoir data - only on 1st, 8th, 15th, and 22nd of each month at 6:00 and 18:00.

        Default: append from the day after the last recorded date up to end_date (now).
        rebuild=True: rebuild full history from start_date (default 2020) up to end_date (now).
        """
        end_date = end_date or self.default_end_date

        # Determine start: from the latest existing date (append) or from scratch (rebuild)
        append_start = None
        if rebuild:
            file_exists = False
            base_start = start_date or self.default_start_date
            effective_start = base_start.replace(hour=6, minute=0, second=0, microsecond=0)
        else:
            append_start, file_exists = self._detect_append_start(self.water_output, "date")
            if append_start is not None:
                effective_start = max(start_date, append_start) if start_date else append_start
            else:
                base_start = start_date or self.default_start_date
                effective_start = base_start.replace(hour=6, minute=0, second=0, microsecond=0)

        if effective_start > end_date:
            last_str = (append_start - timedelta(days=1)).strftime('%Y-%m-%d') if append_start else 'N/A'
            print(f"✅ Water data already up to date (last date: {last_str})")
            return

        start_date = effective_start
        
        # Generate dates for specific days of each month (1st, 8th, 15th, 22nd) at 6:00 and 18:00
        dates = []
        current_date = start_date.replace(day=1)  # Start from 1st of the month
        target_days = [1, 8, 15, 22]
        target_times = [6, 18]  # 6:00 AM and 6:00 PM
        
        while current_date <= end_date:
            for day in target_days:
                for hour in target_times:
                    try:
                        # Create date for specific day and time of current month
                        target_date = current_date.replace(day=day, hour=hour, minute=0, second=0, microsecond=0)
                        if start_date <= target_date <= end_date:
                            dates.append(target_date)
                    except ValueError:
                        # Skip if day doesn't exist in this month (e.g., February 30th)
                        continue
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)

        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5, ssl=False)
        timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=30)
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        ) as session:
            batch_size = 20
            total_rows = 0
            semaphore = asyncio.Semaphore(5)
            
            async def fetch_with_semaphore(date):
                async with semaphore:
                    await asyncio.sleep(0.5)
                    return await self.fetch_water_data(session, date)
            
            write_mode = "a" if file_exists else "w"
            with open(self.water_output, mode=write_mode, newline="", encoding=self.csv_encoding) as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(self.water_header)
                
                for i in range(0, len(dates), batch_size):
                    batch_dates = dates[i:i + batch_size]
                    print(f"🔄 Processing water data batch {i//batch_size + 1}/{(len(dates) + batch_size - 1)//batch_size} (Days: 1st, 8th, 15th, 22nd at 6:00 & 18:00)")
                    
                    tasks = [fetch_with_semaphore(date) for date in batch_dates]
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, Exception):
                            continue
                        for row in result:
                            writer.writerow(row)
                            total_rows += 1
                    
                    if i + batch_size < len(dates):
                        await asyncio.sleep(2)

        self._dedupe_csv(self.water_output, ["date", "region", "reservoir_name"])
        print(f"✅ Water data {'appended to' if file_exists else 'saved to'} {self.water_output}")

    def validate_water_data(self):
        """Validate water reservoir data and report missing regions/years"""
        try:
            df = pd.read_csv(self.water_output, encoding=self.csv_encoding)
            print("🔍 Validating water reservoir data...")
            
            # Parse dates
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
            df['year'] = df['date'].dt.year
            
            # Expected regions and years
            expected_regions = ['Tây Nguyên', 'Tây Bắc Bộ', 'Đông Bắc Bộ', 'Bắc Trung Bộ', 'Nam Trung Bộ', 'Đông Nam Bộ']
            expected_years = list(range(2020, 2026))  # 2020-2025
            
            # Check regions
            available_regions = df['region'].unique()
            missing_regions = [r for r in expected_regions if r not in available_regions]
            
            print(f"📊 Total records: {len(df):,}")
            print(f"🌍 Regions found: {len(available_regions)} / {len(expected_regions)}")
            
            for region in available_regions:
                region_data = df[df['region'] == region]
                years = sorted(region_data['year'].unique())
                reservoirs = region_data['reservoir_name'].nunique()
                print(f"  ✅ {region}: {len(region_data):,} records, {reservoirs} reservoirs, years {years[0] if years else 'N/A'}-{years[-1] if years else 'N/A'}")
            
            if missing_regions:
                print(f"❌ Missing regions: {', '.join(missing_regions)}")
                
            # Check for missing years in critical regions
            critical_regions = ['Tây Nguyên', 'Tây Bắc Bộ']
            for region in critical_regions:
                if region in available_regions:
                    region_data = df[df['region'] == region]
                    available_years = sorted(region_data['year'].unique())
                    missing_years = [y for y in expected_years if y not in available_years]
                    if missing_years:
                        print(f"⚠️ {region} missing data for years: {missing_years}")
                    
                    # Check 2020-2021 specifically
                    early_years = [y for y in available_years if y in [2020, 2021]]
                    if not early_years:
                        print(f"❌ {region} missing critical early data (2020-2021)")
                    else:
                        print(f"✅ {region} has early data for: {early_years}")
            
            return {
                'total_records': len(df),
                'available_regions': list(available_regions),
                'missing_regions': missing_regions,
                'years_range': (df['year'].min(), df['year'].max()) if len(df) > 0 else (None, None)
            }
            
        except Exception as e:
            print(f"❌ Error validating water data: {e}")
            return None

    def _detect_append_start(self, output_path, date_col):
        """Return (next_start_date, file_exists) based on the last date in the CSV.

        The start is the day AFTER the last recorded date (exclusive), so the last
        day is never re-fetched. Combined with end_date = yesterday, each run only
        fetches whole, not-yet-recorded days and can never create duplicates.
        Slash-separated dates (dd/mm/yyyy, e.g. the water timestamps) are parsed
        with dayfirst=True; ISO dates (YYYY-MM-DDTHH:MM:SS) are parsed as-is.
        """
        if not os.path.exists(output_path):
            return None, False
        try:
            df = pd.read_csv(output_path, usecols=[date_col], encoding=self.csv_encoding)
            series = df[date_col].dropna()
            if series.empty:
                return None, True
            # dd/mm/yyyy uses slashes; ISO uses dashes/T. Pick dayfirst accordingly.
            # format="mixed" parses each value independently so columns that mix
            # slash and ISO dates (e.g. capacity files) still resolve correctly.
            dayfirst = "/" in str(series.iloc[0])
            last_date = pd.to_datetime(series, dayfirst=dayfirst, format="mixed").max()
            # Day after the last recorded date, normalized to midnight.
            return last_date.normalize() + timedelta(days=1), True
        except Exception:
            return None, False

    def _dedupe_csv(self, output_path, subset):
        """Drop duplicate rows (keep last) after an inclusive append.

        Everything is read as text so untouched rows are rewritten verbatim
        (no numeric/date reformatting). The file is only rewritten if duplicates
        were actually found.
        """
        if not os.path.exists(output_path):
            return
        try:
            df = pd.read_csv(output_path, dtype=str, encoding=self.csv_encoding)
        except Exception:
            return
        if df.empty:
            return
        subset = [c for c in subset if c in df.columns]
        if not subset:
            return
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="last")
        removed = before - len(df)
        if removed:
            df.to_csv(output_path, index=False, encoding=self.csv_encoding)
            print(f"🧹 Removed {removed} duplicate row(s) from {os.path.basename(output_path)}")

    def _iter_daily_dates(self, start_date, end_date):
        current_date = start_date
        while current_date <= end_date:
            yield current_date
            current_date += timedelta(days=1)

    def _iter_monthly_dates(self, start_date, end_date):
        """Yield the last day of each month from start_date to end_date."""
        import calendar
        current = start_date.replace(day=1)
        while current <= end_date:
            last_day = calendar.monthrange(current.year, current.month)[1]
            target = current.replace(day=last_day)
            if target > end_date:
                target = end_date
            yield target
            # advance to first day of next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)

    def _parse_vn_number(self, value):
        if value is None:
            return None
        text = str(value).replace("\xa0", " ").strip()
        if not text:
            return None
        text = text.replace(".", "").replace(",", ".")
        try:
            return float(text)
        except ValueError:
            return None

    def _extract_report_date(self, title_text):
        if not title_text:
            return None
        match = re.search(r"NGÀY\s+(\d{2}/\d{2}/\d{4})", title_text, re.IGNORECASE)
        if not match:
            return None
        try:
            return datetime.strptime(match.group(1), "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            return None

    def _create_capacity_session(self, parent_url):
        """Visit parent page to obtain XSRF token and return a primed session."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        })
        try:
            session.get(parent_url, timeout=30, verify=False)
            raw_token = session.cookies.get("XSRF-TOKEN")
            if raw_token:
                session.headers["x-xsrf-token"] = urllib.parse.unquote(raw_token)
                print(f"✅ XSRF token obtained from {parent_url}")
            else:
                print(f"⚠️ No XSRF-TOKEN cookie found from {parent_url}")
        except Exception as e:
            print(f"⚠️ Could not get XSRF token: {e}")
        return session

    def _fetch_capacity_table_html(self, url, day, session=None, referer=None, referers=None, retries=3, retry_delay_sec=1):
        day_str = day.strftime("%d/%m/%Y")
        referer_candidates = list(referers or [])
        if referer:
            referer_candidates.insert(0, referer)
        if not referer_candidates:
            referer_candidates = ["https://www.nsmo.vn/HeThongDien"]

        # Keep order while removing duplicates.
        seen = set()
        referer_candidates = [r for r in referer_candidates if not (r in seen or seen.add(r))]

        last_error = None
        for referer_value in referer_candidates:
            request_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                "Accept": "text/plain, */*; q=0.01",
                "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Content-Type": "application/text; charset=utf-8",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": referer_value,
                "sec-ch-ua": '"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
            }

            for attempt in range(retries):
                try:
                    if session is not None:
                        response = session.get(
                            url,
                            params={"day": day_str},
                            headers=request_headers,
                            timeout=30,
                            verify=False,
                        )
                    else:
                        response = requests.get(
                            url,
                            params={"day": day_str},
                            headers=request_headers,
                            timeout=30,
                            verify=False,
                        )
                    response.raise_for_status()
                    return self._decode_http_response_text(response)
                except Exception as e:
                    last_error = e
                    if attempt < retries - 1:
                        time.sleep(retry_delay_sec)

        raise last_error

    def _parse_capacity_table_rows(self, html):
        soup = BeautifulSoup(html, "html.parser")
        title_text = soup.get_text(" ", strip=True)
        report_date = self._extract_report_date(title_text)

        rows = []
        for tr in soup.find_all("tr"):
            cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cols = [col for col in cols if col != ""]
            if len(cols) < 2:
                continue

            source = self._repair_mojibake_text(cols[0])
            numeric_values = [self._parse_vn_number(col) for col in cols[1:]]
            if all(v is None for v in numeric_values):
                continue

            rows.append({
                "source": source,
                "values": numeric_values
            })

        return report_date, rows

    def _append_capacity_rows(self, output_path, header, rows, key_cols):
        if not rows:
            return 0

        df_new = pd.DataFrame(rows)
        if os.path.exists(output_path):
            df_old = pd.read_csv(output_path, encoding=self.csv_encoding)
            for col in header:
                if col not in df_old.columns:
                    df_old[col] = None
            df_old = df_old[header]
            df_all = pd.concat([df_old, df_new[header]], ignore_index=True)
            before = len(df_all)
            df_all = df_all.drop_duplicates(subset=key_cols, keep="last")
            added = len(df_all) - len(df_old)
        else:
            df_all = df_new[header].drop_duplicates(subset=key_cols, keep="last")
            added = len(df_all)

        df_all.to_csv(output_path, index=False, encoding=self.csv_encoding)
        return added

    def scrape_capacity_mobilized_source_data(self, start_date=None, end_date=None, max_workers=20, monthly=False, rebuild=False):
        """Fetch công suất huy động theo nguồn theo ngày từ trang HeThongDien – appends from the last existing date."""
        end_date = end_date or self.default_end_date

        # Detect last existing date and adjust start (or rebuild full history)
        append_start = None
        if rebuild:
            effective_start = start_date or self.default_start_date
        else:
            append_start, _ = self._detect_append_start(self.capacity_mobilized_output, "date")
            if append_start is not None:
                effective_start = max(start_date, append_start) if start_date else append_start
            else:
                effective_start = start_date or self.default_start_date

        if effective_start > end_date:
            last_str = (append_start - timedelta(days=1)).strftime('%Y-%m-%d') if append_start else 'N/A'
            print(f"✅ HeThongDien mobilized source data already up to date (last date: {last_str})")
            return
        start_date = effective_start

        parent_url = "https://www.nsmo.vn/HeThongDien"
        init_session = self._create_capacity_session(parent_url)
        # Snapshot cookies so each thread can create its own session (thread-safe)
        cookies_snapshot = {c.name: c.value for c in init_session.cookies}
        xsrf_token = init_session.headers.get("x-xsrf-token", "")

        if monthly:
            dates = list(self._iter_monthly_dates(start_date, end_date))
            print(f"📅 Monthly mode: {len(dates)} months to fetch")
        else:
            dates = list(self._iter_daily_dates(start_date, end_date))

        def fetch_day(day):
            try:
                s = requests.Session()
                s.headers.update({
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
                    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
                })
                for name, value in cookies_snapshot.items():
                    s.cookies.set(name, value)
                if xsrf_token:
                    s.headers["x-xsrf-token"] = xsrf_token
                html = self._fetch_capacity_table_html(
                    self.capacity_mobilized_url, day,
                    session=s, referer=parent_url
                )
                report_date, parsed_rows = self._parse_capacity_table_rows(html)
                rows = []
                for item in parsed_rows:
                    values = item["values"]
                    rows.append({
                        "date": report_date,
                        "source": item["source"],
                        "low_season": values[0] if len(values) >= 1 else None,
                        "high_season": values[1] if len(values) >= 2 else None,
                    })
                print(f"✅ HTD {day.strftime('%Y-%m-%d')}: parsed={len(rows)}")
                return rows
            except Exception as e:
                print(f"❌ HTD {day.strftime('%Y-%m-%d')}: {e}")
                return []

        all_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for rows in executor.map(fetch_day, dates):
                all_rows.extend(rows)

        added = self._append_capacity_rows(
            self.capacity_mobilized_output,
            self.capacity_mobilized_header,
            all_rows,
            key_cols=["date", "source"]
        )
        print(f"✅ HeThongDien mobilized source data saved to {self.capacity_mobilized_output}; total new rows: {added}")

    def scrape_capacity_installed_source_data(self, start_date=None, end_date=None, max_workers=20, monthly=False, rebuild=False):
        """Fetch tổng công suất lắp đặt theo nguồn theo ngày từ trang ThiTruongDien – appends from the last existing date."""
        end_date = end_date or self.default_end_date

        # Detect last existing date and adjust start (or rebuild full history)
        append_start = None
        if rebuild:
            effective_start = start_date or self.default_start_date
        else:
            append_start, _ = self._detect_append_start(self.capacity_installed_output, "date")
            if append_start is not None:
                effective_start = max(start_date, append_start) if start_date else append_start
            else:
                effective_start = start_date or self.default_start_date

        if effective_start > end_date:
            last_str = (append_start - timedelta(days=1)).strftime('%Y-%m-%d') if append_start else 'N/A'
            print(f"✅ ThiTruongDien installed source data already up to date (last date: {last_str})")
            return
        start_date = effective_start

        parent_url = "https://www.nsmo.vn/ThiTruongDien"
        if monthly:
            dates = list(self._iter_monthly_dates(start_date, end_date))
            print(f"📅 Monthly mode: {len(dates)} months to fetch")
            max_workers = min(max_workers, 6)
        else:
            dates = list(self._iter_daily_dates(start_date, end_date))
        installed_referers = [
            "https://www.nsmo.vn/ThiTruongDien",
            "https://www.nsmo.vn/HeThongDien",
        ]

        def fetch_day(day):
            try:
                # Use a fresh primed session per day to avoid stale token/cookie snapshots
                # that can make historical queries return repeated datasets.
                s = self._create_capacity_session(parent_url)
                html = self._fetch_capacity_table_html(
                    self.capacity_installed_url,
                    day,
                    session=s,
                    referers=installed_referers,
                    retries=3,
                )
                report_date, parsed_rows = self._parse_capacity_table_rows(html)
                if not parsed_rows:
                    html = self._fetch_capacity_table_html(
                        self.capacity_installed_url,
                        day,
                        session=None,
                        referers=installed_referers,
                        retries=2,
                    )
                    report_date, parsed_rows = self._parse_capacity_table_rows(html)

                # Guard against stale/redirected payloads where report_date is unrelated
                # to the requested day.
                if report_date:
                    try:
                        report_dt = datetime.strptime(report_date, "%Y-%m-%d")
                        if abs((report_dt.date() - day.date()).days) > 2:
                            print(
                                f"⚠️ TTD {day.strftime('%Y-%m-%d')}: stale report_date={report_date}, retrying with no session"
                            )
                            html = self._fetch_capacity_table_html(
                                self.capacity_installed_url,
                                day,
                                session=None,
                                referers=installed_referers,
                                retries=2,
                            )
                            report_date, parsed_rows = self._parse_capacity_table_rows(html)
                    except ValueError:
                        pass

                query_date = day.strftime("%Y-%m-%d")
                rows = []
                for item in parsed_rows:
                    values = item["values"]
                    rows.append({
                        "date": query_date,
                        "source": item["source"],
                        "installed_total_mw": values[0] if len(values) >= 1 else None,
                        "low_peak_mw": values[1] if len(values) >= 2 else None,
                        "high_peak_mw": values[2] if len(values) >= 3 else None,
                    })
                print(f"✅ TTD {query_date}: parsed={len(rows)}")
                return rows
            except Exception as e:
                print(f"❌ TTD {day.strftime('%Y-%m-%d')}: {e}")
                return []

        all_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for rows in executor.map(fetch_day, dates):
                all_rows.extend(rows)

        added = self._append_capacity_rows(
            self.capacity_installed_output,
            self.capacity_installed_header,
            all_rows,
            key_cols=["date", "source"]
        )
        print(f"✅ ThiTruongDien installed source data saved to {self.capacity_installed_output}; total new rows: {added}")

    async def run_all_scrapers(self, start_date=None, end_date=None, rebuild=False):
        """Run all scrapers in sequence.

        By default every scraper appends only the new dates after the last record
        already stored in its CSV (set rebuild=True to re-download full history).
        """
        print("🚀 Starting comprehensive data scraping...")

        # Run scrapers in sequence to avoid overwhelming servers
        await self.scrape_price_data(start_date, end_date)
        await asyncio.sleep(2)

        await self.scrape_volume_data(start_date, end_date)
        await asyncio.sleep(2)

        self.scrape_power_capacity_monthly_data(start_date, end_date, rebuild=rebuild)
        await asyncio.sleep(2)

        await self.scrape_water_data(start_date, end_date, rebuild=rebuild)
        await asyncio.sleep(2)

        # Scrape Vinacomin commodity data (synchronous)
        self.scrape_vinacomin_data()

        # Scrape temperature data (synchronous, Open-Meteo API)
        self.scrape_temperature_data(start_date, end_date)

        # Scrape source-capacity tables from HeThongDien and ThiTruongDien (synchronous)
        self.scrape_capacity_mobilized_source_data(start_date, end_date, rebuild=rebuild)
        self.scrape_capacity_installed_source_data(start_date, end_date, rebuild=rebuild)

        # Scrape hydrology current monthly data (RONI)
        self.scrape_hydrology_current_monthly_data(start_date, end_date)
        self.scrape_hydrology_forecast_monthly_data(start_date, end_date)
        
        # Process weighted averages
        print("📊 Processing weighted averages...")
        self.process_weighted_averages()
        
        print("✅ All data processing completed!")

def main():
    # Emoji/Vietnamese log lines crash on non-UTF-8 consoles (Windows cp1252),
    # especially when stdout is redirected to a file. Force UTF-8 with a safe
    # fallback so logging never aborts the scrape.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description='Combined Data Processing Tool')
    parser.add_argument('--action', choices=['price', 'volume', 'pmax', 'renewable', 'water', 'vinacomin', 'temperature', 'hydrology_current', 'hydrology_forecast', 'htd_huydong', 'ttd_lapdat', 'capacity_pages', 'match', 'all'], 
                       default='all', help='Action to perform')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--monthly', action='store_true', help='Fetch one data point per month (last day of each month)')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild output file from requested date range (overwrite file)')
    
    args = parser.parse_args()
    
    processor = DataProcessor()
    
    # Parse dates if provided
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Run the appropriate action
    if args.action == 'price':
        asyncio.run(processor.scrape_price_data(start_date, end_date))
    elif args.action == 'volume':
        asyncio.run(processor.scrape_volume_data(start_date, end_date))
    elif args.action == 'pmax' or args.action == 'renewable':  # Support both names for backward compatibility
        processor.scrape_power_capacity_monthly_data(start_date, end_date, monthly=args.monthly, rebuild=args.rebuild)
    elif args.action == 'water':
        asyncio.run(processor.scrape_water_data(start_date, end_date, rebuild=args.rebuild))
    elif args.action == 'vinacomin':
        processor.scrape_vinacomin_data()
    elif args.action == 'temperature':
        processor.scrape_temperature_data(start_date, end_date)
    elif args.action == 'hydrology_current':
        processor.scrape_hydrology_current_monthly_data(start_date, end_date)
    elif args.action == 'hydrology_forecast':
        processor.scrape_hydrology_forecast_monthly_data(start_date, end_date)
    elif args.action == 'htd_huydong':
        processor.scrape_capacity_mobilized_source_data(start_date, end_date, monthly=args.monthly, rebuild=args.rebuild)
    elif args.action == 'ttd_lapdat':
        processor.scrape_capacity_installed_source_data(start_date, end_date, monthly=args.monthly, rebuild=args.rebuild)
    elif args.action == 'capacity_pages':
        processor.scrape_capacity_mobilized_source_data(start_date, end_date, monthly=args.monthly, rebuild=args.rebuild)
        processor.scrape_capacity_installed_source_data(start_date, end_date, monthly=args.monthly, rebuild=args.rebuild)
    elif args.action == 'match':
        processor.process_weighted_averages()
    elif args.action == 'all':
        asyncio.run(processor.run_all_scrapers(start_date, end_date, rebuild=args.rebuild))

if __name__ == "__main__":
    main()
