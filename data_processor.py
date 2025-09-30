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
import ssl
import certifi
import urllib.parse
import argparse
import time
import requests
import json
import os

class DataProcessor:
    def __init__(self):
        # Configuration
        self.default_start_date = datetime(2020, 1, 1)
        self.default_end_date = datetime.now()
        
        # Get dashboard directory path
        self.dashboard_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Output files (with full paths to dashboard/data folder to overwrite existing files)
        self.price_output = os.path.join(self.dashboard_dir, "data", "cgm_price_monthly.csv")
        self.volume_output = os.path.join(self.dashboard_dir, "data", "power_load_monthly.csv")
        self.pmax_output = os.path.join(self.dashboard_dir, "data", "p_max_monthly.csv")
        self.water_output = os.path.join(self.dashboard_dir, "data", "water_reservoir_monthly.csv")
        self.weighted_output = os.path.join(self.dashboard_dir, "data", "average_prices_monthly.csv")
        self.vinacomin_output = os.path.join(self.dashboard_dir, "data", "vinacomin_data_monthly.csv")
        
        # API URLs
        self.price_api_url = "https://www.nsmo.vn/api/services/app/Pages/GetChartGiaBienVM"
        self.volume_api_url = "https://www.nsmo.vn/api/services/app/Pages/GetChartPhuTaiVM"
        self.pmax_base_url = "https://www.nsmo.vn/HTDThongSoVH"
        self.water_base_url = "https://hochuathuydien.evn.com.vn/PageHoChuaThuyDienEmbedEVN.aspx"
        self.vinacomin_api_url = "https://vinacomin.vn/api/app/commodity/chart"
        
        # Headers
        self.price_header = ["thoiGianCapNhat", "thoiGian", "giaBienMB", "giaBienMT", "giaBienMN", "giaBienHT"]
        self.volume_header = ["thoiGianCapNhat", "thoiGian", "congSuatMB", "congSuatMT", "congSuatMN", "congSuatHT"]
        self.pmax_header = [
            "date", "max_power_thuong_pham_MW", "max_power_thuong_pham_time", "generation_thuong_pham_mkWh",
            "max_power_dau_cuc_MW", "max_power_dau_cuc_time", "generation_dau_cuc_mkWh",
            "thuy_dien_mkWh", "nhiet_dien_than_mkWh", "tuabin_khi_mkWh", "nhiet_dien_dau_mkWh",
            "dien_gio_mkWh", "dmt_trang_trai_mkWh", "dmt_mai_thuong_pham_mkWh", "dmt_mai_dau_cuc_mkWh",
            "nhap_khau_mkWh", "khac_mkWh"
        ]
        self.water_header = ["date_time", "region", "reservoir_name", "flood_level", "flood_capacity", "plant_throughput"]

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
    async def fetch_renewable_html(self, session, date):
        date_str = date.strftime("%d/%m/%Y")
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
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
            
            print(f"Successfully parsed renewable data for {date.strftime('%Y-%m-%d')}: {len(result)} fields")
            return result
            
        except Exception as e:
            print(f"Parsing error for renewable data on {date}: {e}")
            return None

    # Water Reservoir Data Scraping
    async def fetch_water_data(self, session, date, max_retries=3):
        date_str = date.strftime("%d/%m/%Y %H:%M")
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
                                            date_str, 
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
                    color = commodity.get('color', 'DefaultColor')
                    
                    # Process each price record for this commodity
                    for price_data in commodity.get('prices', []):
                        record = {
                            'commodity': commodity_name,
                            'color': color,
                            'update_date': price_data.get('updateDate'),
                            'week': price_data.get('week'),
                            'price': price_data.get('price'),
                            'currency_per_unit': price_data.get('currencyPerUnit'),
                            'percent_change': price_data.get('percentChange'),
                            'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        processed_records.append(record)
                
                # Create DataFrame
                df = pd.DataFrame(processed_records)
                
                # Convert data types and process
                if not df.empty:
                    df['update_date'] = pd.to_datetime(df['update_date'])
                    df['week'] = df['week'].astype(int)
                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                    df['percent_change'] = pd.to_numeric(df['percent_change'], errors='coerce')
                    
                    # Sort by commodity and week
                    df = df.sort_values(['commodity', 'week']).reset_index(drop=True)
                    
                    # Save to CSV
                    df.to_csv(self.vinacomin_output, index=False, encoding='utf-8')
                    
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

    # Data Processing and Matching
    def process_weighted_averages(self):
        """Process volume and price data to calculate weighted averages"""
        try:
            print("Processing weighted average prices...")
            
            # Read the data files
            volume_df = pd.read_csv(self.volume_output)
            price_df = pd.read_csv(self.price_output)
            
            # Convert datetime columns
            volume_df['thoiGian'] = pd.to_datetime(volume_df['thoiGian'])
            price_df['thoiGian'] = pd.to_datetime(price_df['thoiGian'])
            
            # Remove MB, MT, MN columns from volume data
            volume_df = volume_df[['thoiGian', 'congSuatHT']]
            
            # Merge the dataframes on datetime
            merged_df = pd.merge(volume_df, 
                               price_df[['thoiGian', 'giaBienHT']], 
                               on='thoiGian', 
                               how='inner')
            
            # Remove rows where price < 50
            merged_df = merged_df[merged_df['giaBienHT'] >= 50]
            
            # Add date column for grouping
            merged_df['date'] = merged_df['thoiGian'].dt.date
            
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
            result_df.to_csv(self.weighted_output, index=False)
            print(f"Weighted average prices saved to {self.weighted_output}")
            
            return result_df
            
        except Exception as e:
            print(f"Error processing weighted averages: {str(e)}")
            return None

    async def scrape_price_data(self, start_date=None, end_date=None):
        """Scrape price data"""
        start_date = start_date or self.default_start_date
        end_date = end_date or self.default_end_date
        
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
            
            with open(self.price_output, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.price_header)
                
                for result in results:
                    if result:
                        for item in result["giaBiens"]:
                            writer.writerow([
                                result["thoiGianCapNhat"],
                                item.get("thoiGian"),
                                item.get("giaBienMB"),
                                item.get("giaBienMT"),
                                item.get("giaBienMN"),
                                item.get("giaBienHT"),
                            ])

        print(f"✅ Price data saved to {self.price_output}")

    async def scrape_volume_data(self, start_date=None, end_date=None):
        """Scrape volume data"""
        start_date = start_date or self.default_start_date
        end_date = end_date or self.default_end_date
        
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
            
            with open(self.volume_output, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.volume_header)
                
                for result in results:
                    if result:
                        for item in result["phuTais"]:
                            writer.writerow([
                                result["thoiGianCapNhat"],
                                item.get("thoiGian"),
                                item.get("congSuatMB"),
                                item.get("congSuatMT"),
                                item.get("congSuatMN"),
                                item.get("congSuatHT"),
                            ])

        print(f"✅ Volume data saved to {self.volume_output}")

    async def scrape_p_max_data(self, start_date=None, end_date=None):
        """Scrape P Max (Maximum Power) data - default from June 2024"""
        start_date = start_date or datetime(2024, 6, 1)  # Default start from June 2024
        end_date = end_date or self.default_end_date
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        semaphore = asyncio.Semaphore(5)  # Reduced concurrency for better reliability
        all_rows = []

        async def fetch_and_parse(session, date):
            async with semaphore:
                await asyncio.sleep(1)  # Add delay between requests
                html = await self.fetch_renewable_html(session, date)
                if html:
                    result = self.parse_renewable_html(html, date)
                    if result:
                        return result
                return None

        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5, ssl=False)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            print(f"🔄 Scraping P Max data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Process in smaller batches
            batch_size = 50
            for i in range(0, len(dates), batch_size):
                batch_dates = dates[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(dates) + batch_size - 1)//batch_size}")
                
                tasks = [fetch_and_parse(session, d) for d in batch_dates]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for r in results:
                    if r and not isinstance(r, Exception):
                        all_rows.append(r)
                
                # Add delay between batches
                if i + batch_size < len(dates):
                    await asyncio.sleep(5)

        # Sort rows by date
        all_rows.sort(key=lambda x: x[0])

        with open(self.pmax_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.pmax_header)
            writer.writerows(all_rows)
        
        print(f"✅ P Max data saved to {self.pmax_output}")
        print(f"📊 Total records saved: {len(all_rows)}")
        if all_rows:
            print(f"📅 Date range: {all_rows[0][0]} to {all_rows[-1][0]}")

    async def scrape_water_data(self, start_date=None, end_date=None):
        """Scrape water reservoir data - only on 1st, 8th, 15th, and 22nd of each month at 6:00 and 18:00"""
        start_date = start_date or datetime(2020, 1, 1, 6, 0)
        end_date = end_date or self.default_end_date
        
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
            
            with open(self.water_output, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
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

        print(f"✅ Water data saved to {self.water_output}")

    def validate_water_data(self):
        """Validate water reservoir data and report missing regions/years"""
        try:
            df = pd.read_csv(self.water_output)
            print("🔍 Validating water reservoir data...")
            
            # Parse dates
            df['date_time'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M')
            df['year'] = df['date_time'].dt.year
            
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

    async def run_all_scrapers(self, start_date=None, end_date=None):
        """Run all scrapers in sequence"""
        print("🚀 Starting comprehensive data scraping...")
        
        # Run scrapers in sequence to avoid overwhelming servers
        await self.scrape_price_data(start_date, end_date)
        await asyncio.sleep(2)
        
        await self.scrape_volume_data(start_date, end_date)
        await asyncio.sleep(2)
        
        await self.scrape_p_max_data(start_date, end_date)
        await asyncio.sleep(2)
        
        await self.scrape_water_data(start_date, end_date)
        await asyncio.sleep(2)
        
        # Scrape Vinacomin commodity data (synchronous)
        self.scrape_vinacomin_data()
        
        # Process weighted averages
        print("📊 Processing weighted averages...")
        self.process_weighted_averages()
        
        print("✅ All data processing completed!")

def main():
    parser = argparse.ArgumentParser(description='Combined Data Processing Tool')
    parser.add_argument('--action', choices=['price', 'volume', 'pmax', 'renewable', 'water', 'vinacomin', 'match', 'all'], 
                       default='all', help='Action to perform')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
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
        asyncio.run(processor.scrape_p_max_data(start_date, end_date))
    elif args.action == 'water':
        asyncio.run(processor.scrape_water_data(start_date, end_date))
    elif args.action == 'vinacomin':
        processor.scrape_vinacomin_data()
    elif args.action == 'match':
        processor.process_weighted_averages()
    elif args.action == 'all':
        asyncio.run(processor.run_all_scrapers(start_date, end_date))

if __name__ == "__main__":
    main()
