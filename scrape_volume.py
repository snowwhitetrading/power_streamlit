import asyncio
import aiohttp
import csv
from datetime import datetime, timedelta

# --- CONFIG ---
start_date = datetime(2020, 1, 1)  # YYYY, M, D
end_date = datetime.now()          # Use current date
output_csv = "phutai_data.csv"

# API URL
API_URL = "https://www.nsmo.vn/api/services/app/Pages/GetChartPhuTaiVM"

# Prepare CSV header
header = [
    "thoiGianCapNhat",
    "thoiGian",
    "congSuatMB",
    "congSuatMT",
    "congSuatMN",
    "congSuatHT"
]

async def fetch_data(session, date):
    date_str = date.strftime("%d/%m/%Y")
    print(f"Fetching data for {date_str}...")
    
    try:
        async with session.get(API_URL, params={"day": date_str}, ssl=False) as response:
            if response.status != 200:
                print(f"Failed to fetch {date_str}: HTTP {response.status}")
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
                print(f"No data for {date_str}")
                return None
    except Exception as e:
        print(f"Error fetching {date_str}: {str(e)}")
        return None

async def main():
    # Generate all dates
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(50)

    async def fetch_with_semaphore(session, date):
        async with semaphore:
            return await fetch_data(session, date)

    # Create session for all requests
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_semaphore(session, date) for date in dates]
        results = await asyncio.gather(*tasks)
        
        # Write results to CSV
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
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

    print(f"✅ Data saved to {output_csv}")

# Run the async program
if __name__ == "__main__":
    asyncio.run(main())
