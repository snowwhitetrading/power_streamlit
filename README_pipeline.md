# Vietnamese Power Company Financial Data Pipeline

## Overview

This pipeline automatically downloads, processes, and updates financial data for Vietnamese power companies (REE, PC1, HDG, GEG, POW) by:

1. **Downloading PDFs** from company investor relations websites
2. **Extracting financial data** using OpenAI's structured outputs
3. **Converting quarterly data** from Vietnamese formats to standardized quarters
4. **Updating CSV files** with the extracted financial information

## Features

### 🏢 Supported Companies
- **REE**: REE Corporation - renewable energy
- **PC1**: PC1 Group - renewable energy  
- **HDG**: Hado Group - renewable energy
- **GEG**: Geleximco Energy - renewable energy
- **POW**: PV Power - thermal and renewable energy

### 📊 Data Processing
- **Vietnamese Quarter Recognition**: Automatically converts "Quý I", "6 tháng", "9 tháng", "cả năm" to 1Q, 2Q, 3Q, 4Q
- **Cumulative Data Adjustment**: Converts cumulative figures to quarterly figures
- **Structured Data Extraction**: Uses OpenAI to extract financial metrics from Vietnamese PDFs
- **CSV Integration**: Updates existing company_*_monthly.csv files

### 🔍 Extracted Metrics
- Revenue (Doanh thu)
- Gross Profit (Lợi nhuận gộp)  
- Operating Profit (Lợi nhuận hoạt động)
- Net Profit (Lợi nhuận sau thuế)
- EBITDA
- Total Assets (Tổng tài sản)
- Total Equity (Vốn chủ sở hữu)
- Power Generation (GWh)
- Installed Capacity (MW)

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements_pipeline.txt
```

### 2. Set Up OpenAI API Key
Create a `.env` file or set environment variable:
```bash
# Option 1: .env file
OPENAI_API_KEY=your_openai_api_key_here

# Option 2: Environment variable (Windows)
set OPENAI_API_KEY=your_openai_api_key_here

# Option 3: Environment variable (Linux/Mac)
export OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Quick Start - Run Complete Pipeline
```bash
python financial_data_pipeline.py
```

### Test the System
```bash
python test_pipeline.py
```

### Individual Components

#### 1. Download PDFs Only
```python
from company_pdf_downloader import CompanyPDFDownloader

downloader = CompanyPDFDownloader()
downloads = downloader.download_all_companies()
```

#### 2. Process PDFs Only
```python
from pdf_processor import CompanyPDFProcessor

processor = CompanyPDFProcessor(api_key="your_key")
financial_data = processor.process_pdf("path/to/pdf", "REE")
```

#### 3. Update CSVs Only
```python
from csv_updater import CompanyDataUpdater

updater = CompanyDataUpdater()
results = updater.update_all_companies(financial_data_list)
```

## File Structure

```
dashboard/
├── company_pdf_downloader.py    # PDF scraping and downloading
├── pdf_processor.py             # LangChain PDF processing + OpenAI extraction
├── csv_updater.py              # CSV file updates
├── financial_data_pipeline.py   # Main orchestrator
├── test_pipeline.py            # Test suite
├── requirements_pipeline.txt    # Dependencies
├── search/
│   └── pdfs/                   # Downloaded PDF files
└── data/
    ├── company_ree_monthly.csv # REE financial data
    ├── company_pc1_monthly.csv # PC1 financial data
    ├── company_hdg_monthly.csv # HDG financial data
    ├── company_geg_monthly.csv # GEG financial data
    └── company_pow_monthly.csv # POW financial data (created if needed)
```

## Company-Specific Details

### REE Corporation
- **URL**: https://www.reecorp.com/danh-muc-tai-lieu/cong-bo-thong-tin/?tu-khoa=kinh+doanh&nam=
- **Target**: PDFs with title "Kết quả kinh doanh ..."
- **Data**: Solar, Wind, Hydro, Thermal volumes and revenues

### PC1 Group  
- **URL**: https://www.pc1group.vn/category/quan-he-dau-tu/cong-bo-thong-tin/
- **Target**: PDFs with title "Bản tin nhà đầu tư ..."
- **Data**: Wind and Hydro volumes and revenues

### HDG (Hado Group)
- **URL**: https://hado.com.vn/quan-he-co-dong
- **Target**: PDFs with title "Bản tin nhà đầu tư ..."  
- **Data**: Solar, Wind, Hydro volumes and revenues

### GEG (Geleximco Energy)
- **URL**: https://geccom.vn/thong-cao-bao-chi/thong-cao-bao-chi-ir
- **Target**: Secondary URLs with "Kết quả kinh doanh..."
- **Data**: Solar, Wind, Hydro, Others revenues

### POW (PV Power)
- **URL**: https://pvpower.vn/vi/tag/thong-tin-tai-lieu-co-dong-20.htm
- **Target**: PDFs with title "Thông cáo nhà đầu tư ..."
- **Data**: Power generation and financial metrics

## Vietnamese Data Conversion Rules

### Quarter Mapping
- **"Quý I" / "3 tháng đầu năm"** → 1Q
- **"Quý II"** → 2Q  
- **"6 tháng"** → 2Q (cumulative, subtract 1Q)
- **"Quý III"** → 3Q
- **"9 tháng"** → 3Q (cumulative, subtract 1Q+2Q)
- **"Quý IV" / "12 tháng" / "cả năm"** → 4Q (cumulative, subtract 1Q+2Q+3Q)

### Financial Terms Translation
- **Doanh thu** = Revenue
- **Lợi nhuận gộp** = Gross Profit
- **Lợi nhuận từ hoạt động kinh doanh** = Operating Profit  
- **Lợi nhuận sau thuế** = Net Profit After Tax
- **Tổng tài sản** = Total Assets
- **Vốn chủ sở hữu** = Total Equity

## Error Handling

The pipeline includes comprehensive error handling:
- **Network timeouts** for PDF downloads
- **PDF parsing errors** with fallback mechanisms
- **OpenAI API failures** with retry logic
- **CSV update errors** with validation
- **Detailed logging** for debugging

## Output

### Console Output
- Real-time progress updates
- Download summaries
- Extraction confidence scores
- Update success/failure status

### Generated Files
- **Downloaded PDFs** in `search/pdfs/`
- **Updated CSV files** in `data/`
- **Pipeline results JSON** with complete execution log
- **Error logs** for debugging

### Sample Output
```
📊 FINANCIAL DATA PIPELINE SUMMARY
=====================================

📥 DOWNLOAD PHASE:
   Total PDFs Downloaded: 8
   - REE: 2 files
   - PC1: 2 files
   - HDG: 2 files
   - GEG: 1 files
   - POW: 1 files

🔍 EXTRACTION PHASE:
   Total Records Extracted: 7
   - REE: 2 records
   - PC1: 2 records
   - HDG: 2 records
   - GEG: 1 records

💾 UPDATE PHASE:
   Companies Updated: 4/4
   - REE: ✅ SUCCESS
   - PC1: ✅ SUCCESS
   - HDG: ✅ SUCCESS
   - GEG: ✅ SUCCESS

🎉 OVERALL STATUS: SUCCESSFUL
```

## Integration with Existing Dashboard

This pipeline integrates seamlessly with your existing Streamlit dashboard:

1. **Data Compatibility**: Updates existing CSV structures
2. **ChatGPT Module**: Can be integrated with `chatgpt_module.py`
3. **Automated Updates**: Can be scheduled to run periodically
4. **Dashboard Refresh**: New data automatically appears in dashboard

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   ```
   Error: OpenAI API key is required
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **PDF Download Failures**
   ```
   Error: Failed to download PDFs
   Solution: Check internet connection and website availability
   ```

3. **Low Extraction Confidence**
   ```
   Warning: Low confidence extraction
   Solution: Check PDF quality and Vietnamese text clarity
   ```

4. **CSV Update Failures**
   ```
   Error: Cannot update CSV
   Solution: Check file permissions and CSV structure
   ```

### Debug Mode
Run with detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] Support for more companies
- [ ] Automated scheduling with cron jobs
- [ ] Email notifications for updates
- [ ] Dashboard integration for manual triggers
- [ ] Historical data validation
- [ ] Multi-language support for other markets

## License

This project is part of the power_streamlit dashboard system.

## Support

For issues or questions, check the test suite output and error logs for detailed debugging information.