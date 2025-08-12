import asyncio
import aiohttp
import csv
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import urllib.parse

# --- CONFIG ---
start_date = datetime(2024, 1, 1, 15, 0)  # Start from 2020, 18:00
end_date = datetime.now()
output_csv = "water_reservoir_simplified.csv"  # Simplified output file

BASE_URL = "https://hochuathuydien.evn.com.vn/PageHoChuaThuyDienEmbedEVN.aspx"

# Simplified header with only 5 columns
header = [
    "date_time",
    "region",
    "reservoir_name",
    "flood_level",
    "total_capacity"
]

async def fetch_data(session, date, max_retries=3):
    date_str = date.strftime("%d/%m/%Y %H:%M")
    encoded_date = urllib.parse.quote(date_str)
    url = f"{BASE_URL}?td={encoded_date}&vm=&lv=&hc="
    
    for attempt in range(max_retries):
        try:
            async with session.get(url, ssl=False) as response:
                if response.status != 200:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                rows = []
                region = None
                for tr in soup.find_all('tr'):
                    # Region header
                    if 'tralter' in tr.get('class', []):
                        td = tr.find('td')
                        strong = td.find('strong') if td else None
                        region = strong.get_text(strip=True) if strong else None
                    else:
                        tds = tr.find_all('td')
                        if len(tds) >= 11 and region:
                            # Reservoir name
                            first_td = tds[0]
                            b_tag = first_td.find('b')
                            reservoir_name = b_tag.get_text(strip=True) if b_tag else ""
                            
                            # Only extract the 5 required columns
                            row = [
                                date_str,                           # date_time
                                region,                             # region
                                reservoir_name,                     # reservoir_name
                                tds[5].get_text(strip=True),       # flood_level (column 6)
                                tds[8].get_text(strip=True)        # total_capacity (column 9)
                            ]
                            rows.append(row)
                return rows
                
        except asyncio.TimeoutError:
            print(f"Timeout on attempt {attempt + 1} for {date_str}")
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** attempt)  # Longer delay for timeouts
                continue
        except Exception as e:
            print(f"Error fetching {date_str} on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
    
    print(f"Failed to fetch {date_str} after {max_retries} attempts")
    return []

async def main():
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"📅 Fetching simplified data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"📊 Total days to process: {len(dates)}")
    print(f"📋 Collecting: {', '.join(header)}")

    # Create session with improved timeout settings
    connector = aiohttp.TCPConnector(
        limit=10,  # Reduced connection limit
        limit_per_host=5,  # Lower per-host limit
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True
    )
    
    # Increased timeout values
    timeout = aiohttp.ClientTimeout(
        total=60,      # Total timeout increased to 60 seconds
        connect=15,    # Connection timeout increased to 15 seconds
        sock_read=30   # Socket read timeout
    )
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    ) as session:
        # Process in smaller batches with delays
        batch_size = 20  # Reduced batch size
        total_rows = 0
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Only 5 concurrent requests
        
        async def fetch_with_semaphore(date):
            async with semaphore:
                await asyncio.sleep(0.5)  # Small delay between requests
                return await fetch_data(session, date)
        
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for i in range(0, len(dates), batch_size):
                batch_dates = dates[i:i + batch_size]
                print(f"🔄 Processing batch {i//batch_size + 1}/{(len(dates) + batch_size - 1)//batch_size} ({len(batch_dates)} days)")
                
                tasks = [fetch_with_semaphore(date) for date in batch_dates]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"❌ Batch error: {result}")
                        continue
                    for row in result:
                        writer.writerow(row)
                        total_rows += 1
                
                print(f"✅ Batch {i//batch_size + 1} completed. Records so far: {total_rows}")
                
                # Delay between batches
                if i + batch_size < len(dates):
                    await asyncio.sleep(2)

    print(f"✅ Simplified water reservoir data saved to {output_csv}")
    print(f"📈 Total records: {total_rows}")

if __name__ == "__main__":
    asyncio.run(main())
