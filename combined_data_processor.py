"""
Combined Data Processing Script
Combines functionality from:
- scrape_price.py (price data scraping)
- scrape_volume.py (volume data scraping)
- scrape_type.py (renewable energy type data scraping)
- scrape_water.py (water reservoir data scraping)
- match.py (data matching and weighted average calculation)
"""

import asyncio
import aiohttp
import csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import ssl
import certifi
import urllib.parse
import argparse
import time

class DataProcessor:
    def __init__(self):
        # Configuration
        self.default_start_date = datetime(2020, 1, 1)
        self.default_end_date = datetime.now()
        
        # Output files
        self.price_output = "gia_bien_data.csv"
        self.volume_output = "phutai_data.csv"
        self.renewable_output = "thong_so_vh.csv"
        self.water_output = "water_reservoir.csv"
        self.weighted_output = "weighted_average_prices.csv"
        
        # API URLs
        self.price_api_url = "https://www.nsmo.vn/api/services/app/Pages/GetChartGiaBienVM"
        self.volume_api_url = "https://www.nsmo.vn/api/services/app/Pages/GetChartPhuTaiVM"
        self.renewable_base_url = "https://www.nsmo.vn/HTDThongSoVH"
        self.water_base_url = "https://hochuathuydien.evn.com.vn/PageHoChuaThuyDienEmbedEVN.aspx"
        
        # Headers
        self.price_header = ["thoiGianCapNhat", "thoiGian", "giaBienMB", "giaBienMT", "giaBienMN", "giaBienHT"]
        self.volume_header = ["thoiGianCapNhat", "thoiGian", "congSuatMB", "congSuatMT", "congSuatMN", "congSuatHT"]
        self.renewable_header = [
            "date", "max_power_thuong_pham_MW", "max_power_thuong_pham_time", "generation_thuong_pham_mkWh",
            "max_power_dau_cuc_MW", "max_power_dau_cuc_time", "generation_dau_cuc_mkWh",
            "thuy_dien_mkWh", "nhiet_dien_than_mkWh", "tuabin_khi_mkWh", "nhiet_dien_dau_mkWh",
            "dien_gio_mkWh", "dmt_trang_trai_mkWh", "dmt_mai_thuong_pham_mkWh", "dmt_mai_dau_cuc_mkWh",
            "nhap_khau_mkWh", "khac_mkWh"
        ]
        self.water_header = ["date_time", "region", "reservoir_name", "flood_level", "total_capacity"]

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
            timeout = aiohttp.ClientTimeout(total=15)
            async with session.get(self.renewable_base_url, params={"day": date_str}, ssl=False, timeout=timeout) as resp:
                if resp.status != 200:
                    return None
                return await resp.text()
        except (asyncio.TimeoutError, aiohttp.ClientError):
            return None
        except Exception as e:
            return None

    def parse_renewable_html(self, html, date):
        try:
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)
            
            # Extract data using regex patterns (simplified version)
            import re
            
            # Extract various power generation data
            patterns = {
                'max_power_thuong_pham': r'Công suất cực đại thương phẩm.*?(\d+(?:\.\d+)?)\s*MW',
                'generation_thuong_pham': r'Sản lượng điện thương phẩm.*?(\d+(?:\.\d+)?)',
                'thuy_dien': r'Thủy điện.*?(\d+(?:\.\d+)?)',
                'nhiet_dien_than': r'Nhiệt điện than.*?(\d+(?:\.\d+)?)',
                'tuabin_khi': r'Tuabin khí.*?(\d+(?:\.\d+)?)',
                'dien_gio': r'Điện gió.*?(\d+(?:\.\d+)?)',
                'nhap_khau': r'Nhập khẩu.*?(\d+(?:\.\d+)?)'
            }
            
            result = [date.strftime("%Y-%m-%d")]
            
            # Extract values or use 0 as default
            for key in self.renewable_header[1:]:
                pattern_key = key.replace('_mkWh', '').replace('_MW', '').replace('_time', '')
                if pattern_key in patterns:
                    match = re.search(patterns[pattern_key], text, re.IGNORECASE)
                    result.append(match.group(1) if match else "0")
                else:
                    result.append("0")
            
            return result
        except Exception as e:
            print(f"Parsing error for renewable data on {date}: {e}")
            return None

    # Water Reservoir Data Scraping
    async def fetch_water_data(self, session, date, max_retries=3):
        date_str = date.strftime("%d/%m/%Y %H:%M")
        encoded_date = urllib.parse.quote(date_str)
        url = f"{self.water_base_url}?td={encoded_date}&vm=&lv=&hc="
        
        for attempt in range(max_retries):
            try:
                async with session.get(url, ssl=False) as response:
                    if response.status != 200:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    rows = []
                    region = None
                    
                    for tr in soup.find_all('tr'):
                        if 'tralter' in tr.get('class', []):
                            td = tr.find('td')
                            strong = td.find('strong') if td else None
                            region = strong.get_text(strip=True) if strong else None
                        else:
                            tds = tr.find_all('td')
                            if len(tds) >= 11 and region:
                                try:
                                    reservoir_name = tds[1].get_text(strip=True)
                                    flood_level = tds[8].get_text(strip=True)
                                    total_capacity = tds[10].get_text(strip=True)
                                    
                                    if reservoir_name and flood_level and total_capacity:
                                        rows.append([
                                            date_str, region, reservoir_name,
                                            flood_level, total_capacity
                                        ])
                                except (IndexError, AttributeError):
                                    continue
                    
                    return rows
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"Error fetching water data for {date_str}: {str(e)}")
                return []
        
        return []

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

    async def scrape_renewable_data(self, start_date=None, end_date=None):
        """Scrape renewable energy type data"""
        start_date = start_date or datetime(2024, 6, 1)  # Default for renewable data
        end_date = end_date or self.default_end_date
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

        semaphore = asyncio.Semaphore(10)
        all_rows = []

        async def fetch_and_parse(session, date):
            async with semaphore:
                html = await self.fetch_renewable_html(session, date)
                if html:
                    result = self.parse_renewable_html(html, date)
                    if result:
                        return result
                return None

        connector = aiohttp.TCPConnector(limit=20, limit_per_host=10, ssl=False)
        timeout = aiohttp.ClientTimeout(total=15)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [fetch_and_parse(session, d) for d in dates]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for r in results:
                if r and not isinstance(r, Exception):
                    all_rows.append(r)

        with open(self.renewable_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.renewable_header)
            writer.writerows(all_rows)
        
        print(f"✅ Renewable data saved to {self.renewable_output}")

    async def scrape_water_data(self, start_date=None, end_date=None):
        """Scrape water reservoir data"""
        start_date = start_date or datetime(2024, 1, 1, 15, 0)
        end_date = end_date or self.default_end_date
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(days=1)

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
                    print(f"🔄 Processing water data batch {i//batch_size + 1}/{(len(dates) + batch_size - 1)//batch_size}")
                    
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

    async def run_all_scrapers(self, start_date=None, end_date=None):
        """Run all scrapers in sequence"""
        print("🚀 Starting comprehensive data scraping...")
        
        # Run scrapers in sequence to avoid overwhelming servers
        await self.scrape_price_data(start_date, end_date)
        await asyncio.sleep(2)
        
        await self.scrape_volume_data(start_date, end_date)
        await asyncio.sleep(2)
        
        await self.scrape_renewable_data(start_date, end_date)
        await asyncio.sleep(2)
        
        await self.scrape_water_data(start_date, end_date)
        
        # Process weighted averages
        print("📊 Processing weighted averages...")
        self.process_weighted_averages()
        
        print("✅ All data processing completed!")

def main():
    parser = argparse.ArgumentParser(description='Combined Data Processing Tool')
    parser.add_argument('--action', choices=['price', 'volume', 'renewable', 'water', 'match', 'all'], 
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
    elif args.action == 'renewable':
        asyncio.run(processor.scrape_renewable_data(start_date, end_date))
    elif args.action == 'water':
        asyncio.run(processor.scrape_water_data(start_date, end_date))
    elif args.action == 'match':
        processor.process_weighted_averages()
    elif args.action == 'all':
        asyncio.run(processor.run_all_scrapers(start_date, end_date))

if __name__ == "__main__":
    main()
