import asyncio
import aiohttp
import csv
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# --- CONFIG ---
start_date = datetime(2024, 6, 1)  # Start from June 1, 2024
end_date = datetime.now()
output_csv = "thong_so_vh.csv"

BASE_URL = "https://www.nsmo.vn/HTDThongSoVH"

# CSV header
header = [
    "date",
    "max_power_thuong_pham_MW",
    "max_power_thuong_pham_time",
    "generation_thuong_pham_mkWh",
    "max_power_dau_cuc_MW",
    "max_power_dau_cuc_time",
    "generation_dau_cuc_mkWh",
    "thuy_dien_mkWh",
    "nhiet_dien_than_mkWh",
    "tuabin_khi_mkWh",
    "nhiet_dien_dau_mkWh",
    "dien_gio_mkWh",
    "dmt_trang_trai_mkWh",
    "dmt_mai_thuong_pham_mkWh",
    "dmt_mai_dau_cuc_mkWh",
    "nhap_khau_mkWh",
    "khac_mkWh"
]

async def fetch_html(session, date):
    date_str = date.strftime("%d/%m/%Y")
    try:
        timeout = aiohttp.ClientTimeout(total=15)  # Reduced timeout for better throughput
        async with session.get(BASE_URL, params={"day": date_str}, ssl=False, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return await resp.text()
    except (asyncio.TimeoutError, aiohttp.ClientError):
        return None
    except Exception as e:
        return None

def parse_html(html, date):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    def extract_between(start_marker, end_marker, search_start=0):
        try:
            start = text.index(start_marker, search_start) + len(start_marker)
            end = text.index(end_marker, start)
            return text[start:end].strip()
        except ValueError:
            return ""
    
    def extract_value_after_marker(marker, search_start=0):
        """Extract numerical value that appears after a marker, handling multi-line format"""
        try:
            marker_pos = text.index(marker, search_start)
            # Look for the next number after the marker
            text_after_marker = text[marker_pos + len(marker):]
            
            # Split by common separators and look for numbers
            parts = text_after_marker.replace('\n', ' ').replace('\t', ' ').split()
            
            for i, part in enumerate(parts):
                # Check if this part looks like a number (with comma as decimal separator)
                if part.replace(',', '.').replace('-', '').replace('.', '').isdigit():
                    # Check if 'triệu' appears in the next few parts
                    next_parts = parts[i:i+3]
                    if any('triệu' in p for p in next_parts):
                        # Convert comma decimal separator to dot for consistent format
                        return part.replace(',', '.')
            
            return ""
        except (ValueError, IndexError):
            return ""

    try:
        # Extract main metrics
        max_tp = extract_between("Công suất lớn nhất trong ngày:", "MW (Lúc").replace(",", "")
        time_tp = extract_between("MW (Lúc", ")").strip()
        gen_tp = extract_between("Sản lượng điện sản xuất và nhập khẩu:", "triệu").replace(",", "")

        max_dc = extract_between("Công suất lớn nhất trong ngày:", "MW (Lúc", text.index("Tính với số liệu ĐMT mái nhà (ước tính đầu cực)")).replace(",", "")
        time_dc = extract_between("MW (Lúc", ")", text.index("Tính với số liệu ĐMT mái nhà (ước tính đầu cực)")).strip()
        gen_dc = extract_between("Sản lượng điện sản xuất và nhập khẩu:", "triệu", text.index("Tính với số liệu ĐMT mái nhà (ước tính đầu cực)")).replace(",", "")

        # Sources breakdown - using improved extraction for multi-line format
        sources = []
        source_markers = [
            "Thủy điện",
            "Nhiệt điện than", 
            "Tuabin khí",
            "Nhiệt điện dầu",
            "Điện gió",
            "ĐMT trang trại",
            "ĐMT mái nhà (ước tính thương phẩm)",
            "ĐMT mái nhà (ước tính đầu cực)",
            "Nhập khẩu điện",
            "Khác"
        ]
        
        for marker in source_markers:
            # Try the new extraction method first
            value = extract_value_after_marker(marker)
            if not value:
                # Fallback to original method with cleaned comma handling
                value = extract_between(marker, "triệu")
                # Preserve decimal format: convert comma to dot for decimal numbers
                if ',' in value and value.replace(',', '').replace('.', '').isdigit():
                    value = value.replace(',', '.')
                else:
                    value = value.replace(',', '')
            sources.append(value)

        return [date.strftime("%Y-%m-%d"), max_tp, time_tp, gen_tp, max_dc, time_dc, gen_dc] + sources

    except Exception as e:
        print(f"Parsing error for {date}: {e}")
        return None

async def main():
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    print(f"📅 Processing {len(dates)} dates from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Reduce semaphore for better stability with large date ranges
    semaphore = asyncio.Semaphore(10)
    processed_count = 0
    total_dates = len(dates)

    async def fetch_and_parse(session, date):
        nonlocal processed_count
        async with semaphore:
            processed_count += 1
            if processed_count % 50 == 0:  # Progress update every 50 requests
                print(f"🔄 Progress: {processed_count}/{total_dates} ({processed_count/total_dates*100:.1f}%)")
            
            html = await fetch_html(session, date)
            if html:
                result = parse_html(html, date)
                if result:
                    return result
                else:
                    print(f"❌ Failed to parse {date.strftime('%Y-%m-%d')}")
            return None

    all_rows = []
    
    # Configure session with connection pooling for better performance
    connector = aiohttp.TCPConnector(
        limit=20,  # Total connection limit
        limit_per_host=10,  # Per-host connection limit
        ssl=False
    )
    timeout = aiohttp.ClientTimeout(total=15)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        print("🌐 Starting to fetch data...")
        tasks = [fetch_and_parse(session, d) for d in dates]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for r in results:
            if r and not isinstance(r, Exception):
                all_rows.append(r)

    print(f"📊 Successfully parsed {len(all_rows)} rows")

    # Save CSV with error handling
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_rows)
        print(f"✅ Data saved to {output_csv}")
    except PermissionError:
        # Try alternative filename
        import time
        timestamp = int(time.time())
        alternative_filename = f"thong_so_vh_{timestamp}.csv"
        try:
            with open(alternative_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(all_rows)
            print(f"⚠️  Original file was locked. Data saved to {alternative_filename}")
        except Exception as e:
            print(f"❌ Error saving file: {e}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
