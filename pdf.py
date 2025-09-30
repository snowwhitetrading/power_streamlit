"""
Vietnamese Power Company Financial Data Pipeline - All-in-One Solution
======================================================================

This comprehensive script handles:
1. PDF downloading from Vietnamese power companies
2. Financial data extraction using OpenAI + LangChain
3. CSV data updates with quarterly conversion
4. Complete pipeline orchestration

Supported Companies: 
- REE (Refrigeration Electrical Engineering Corporation)
- PC1 (Power Construction Corporation 1)
- HDG (Hado Group)
- GEG (Gia Lai Electricity Group)
- POW (PetroVietnam Power Corporation)
"""

import requests
import json
import os
import sys
import re
import pandas as pd
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables first
load_dotenv()

# Install required packages if not present
def install_requirements():
    """Install required packages"""
    required_packages = [
        "langchain",
        "langchain-community", 
        "pypdf",
        "openai",
        "pydantic",
        "beautifulsoup4",
        "pandas",
        "requests"
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

# Install requirements first
install_requirements()

# Now import the packages
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from openai import OpenAI
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run: pip install langchain langchain-community pypdf openai pydantic")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================================================================================
# STRUCTURED DATA MODELS
# ================================================================================================

class QuarterlyFinancialData(BaseModel):
    """Structured model for quarterly financial data"""
    company_code: str = Field(description="Company stock code (REE, PC1, HDG, GEG, POW)")
    year: int = Field(description="Year of the financial data")
    quarter: str = Field(description="Quarter (1Q, 2Q, 3Q, 4Q)")
    
    # Financial metrics (in billion VND unless specified)
    revenue: Optional[float] = Field(None, description="Quarterly revenue in billion VND")
    gross_profit: Optional[float] = Field(None, description="Gross profit in billion VND")
    operating_profit: Optional[float] = Field(None, description="Operating profit (EBIT) in billion VND")
    net_profit: Optional[float] = Field(None, description="Net profit after tax in billion VND")
    ebitda: Optional[float] = Field(None, description="EBITDA in billion VND")
    
    # Additional metrics
    total_assets: Optional[float] = Field(None, description="Total assets in billion VND")
    total_equity: Optional[float] = Field(None, description="Total equity in billion VND")
    cash_and_equivalents: Optional[float] = Field(None, description="Cash and cash equivalents in billion VND")
    
    # Power-specific metrics (if applicable)
    power_generation_gwh: Optional[float] = Field(None, description="Power generation in GWh")
    capacity_mw: Optional[float] = Field(None, description="Installed capacity in MW")
    
    # Metadata
    report_period: str = Field(description="Original report period description (e.g., 'Quý I năm 2025')")
    extraction_confidence: float = Field(description="Confidence score of data extraction (0-1)")

# ================================================================================================
# PDF DOWNLOADER CLASS
# ================================================================================================

class CompanyPDFDownloader:
    def __init__(self, download_dir="pdfs"):
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Company configurations based on specific requirements
        self.companies = {
            'REE': {
                'url': 'https://www.reecorp.com/danh-muc-tai-lieu/cong-bo-thong-tin/?tu-khoa=kinh+doanh&nam=',
                'title_pattern': r'ree.*\d+.*results|ree.*date.*results|ree.*quarterly',
                'method': 'direct_pdf',
                'data_source': 'Năng Lượng pages',
                'csv_file': 'company_ree_monthly.csv'
            },
            'PC1': {
                'url': 'https://www.pc1group.vn/category/quan-he-dau-tu/cong-bo-thong-tin/',
                'title_pattern': r'bản tin nhà đầu tư',
                'method': 'secondary_url',
                'data_source': 'URL content',
                'csv_file': 'company_pc1_monthly.csv'  # Note: User mentioned company_geg_monthly but this should be pc1
            },
            'HDG': {
                'url': 'https://hado.com.vn/quan-he-co-dong',
                'title_pattern': r'ban tin ir.*\d+|bản tin.*ir.*\d+|bản tin nhà đầu tư',
                'method': 'direct_pdf',
                'data_source': 'Năng Lượng page (Solar: Hồng Phong 4, Infra 1; Wind: 7A; Rest: Hydro)',
                'csv_file': 'company_hdg_monthly.csv'
            },
            'GEG': {
                'url': 'https://geccom.vn/thong-cao-bao-chi/thong-cao-bao-chi-ir',
                'title_pattern': r'kết quả kinh doanh',
                'method': 'secondary_url',
                'data_source': 'URL content',
                'csv_file': 'company_geg_monthly.csv'
            },
            'POW': {
                'url': 'https://pvpower.vn/vi/tag/thong-tin-tai-lieu-co-dong-20.htm',
                'title_pattern': r'(thông cáo|nhà đầu tư|báo cáo|tháng)',
                'method': 'direct_pdf',
                'data_source': 'PDF content',
                'csv_file': 'company_pow_monthly.csv'
            }
        }

    def get_page_content(self, url, timeout=10):
        """Get page content with error handling"""
        try:
            response = requests.get(url, headers=self.headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            return None

    def find_pdf_links(self, soup, base_url, title_pattern):
        """Find PDF links matching the title pattern with enhanced debugging"""
        pdf_links = []
        all_links = soup.find_all('a', href=True)
        
        logger.info(f"Found {len(all_links)} total links on page")
        
        # Look for direct PDF links
        pdf_count = 0
        title_matches = 0
        
        for link in all_links:
            href = link['href']
            link_text = link.get_text(strip=True)
            
            # Check if it's a PDF link
            if href.lower().endswith('.pdf') or '.pdf' in href.lower():
                pdf_count += 1
                logger.debug(f"Found PDF link: {href} - Text: {link_text}")
                
                if re.search(title_pattern, link_text, re.IGNORECASE):
                    title_matches += 1
                    full_url = urljoin(base_url, href)
                    # Extract date from element
                    upload_date = self.extract_date_from_element(link)
                    pdf_links.append({
                        'url': full_url,
                        'title': link_text,
                        'element': link,
                        'upload_date': upload_date
                    })
                    date_str = upload_date.strftime('%Y-%m-%d') if upload_date else 'Unknown'
                    logger.info(f"Matched PDF: {link_text} -> {full_url} (Date: {date_str})")
        
        logger.info(f"Found {pdf_count} PDF links, {title_matches} matched title pattern")
        
        # Look for links that might lead to PDFs (for secondary URL method)
        secondary_matches = 0
        for link in all_links:
            link_text = link.get_text(strip=True)
            if re.search(title_pattern, link_text, re.IGNORECASE) and link_text:
                href = link['href']
                if not href.lower().endswith('.pdf'):
                    secondary_matches += 1
                    full_url = urljoin(base_url, href)
                    # Extract date from element
                    upload_date = self.extract_date_from_element(link)
                    pdf_links.append({
                        'url': full_url,
                        'title': link_text,
                        'element': link,
                        'type': 'secondary',
                        'upload_date': upload_date
                    })
                    date_str = upload_date.strftime('%Y-%m-%d') if upload_date else 'Unknown'
                    logger.info(f"Secondary link: {link_text} -> {full_url} (Date: {date_str})")
        
        logger.info(f"Found {secondary_matches} secondary links")
        
        # Sort PDF links by upload date (most recent first)
        # PDFs with dates come first, then those without dates
        pdf_links.sort(key=lambda x: (x['upload_date'] is None, 
                                     -(x['upload_date'].timestamp() if x['upload_date'] else 0)))
        
        if pdf_links:
            logger.info(f"Latest PDF found: {pdf_links[0]['title']} " + 
                       f"(Date: {pdf_links[0]['upload_date'].strftime('%Y-%m-%d') if pdf_links[0]['upload_date'] else 'Unknown'})")
        
        return pdf_links

    def extract_date_from_element(self, element):
        """Extract date from link element or nearby text"""
        # Common Vietnamese date patterns
        date_patterns = [
            r'(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})',  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{4})[\/\-](\d{1,2})[\/\-](\d{1,2})',  # YYYY/MM/DD or YYYY-MM-DD
            r'(\d{1,2})\s*(tháng|th)\s*(\d{1,2})\s*năm\s*(\d{4})',  # Vietnamese format
            r'(\d{4})(\d{2})(\d{2})',  # YYYYMMDD
        ]
        
        # Search in the link text itself
        link_text = element.get_text(strip=True)
        for pattern in date_patterns:
            match = re.search(pattern, link_text, re.IGNORECASE)
            if match:
                try:
                    if 'tháng' in pattern or 'th' in pattern:
                        day, _, month, year = match.groups()
                        return datetime(int(year), int(month), int(day))
                    elif pattern.startswith(r'(\d{4})'):
                        year, month, day = match.groups()
                        return datetime(int(year), int(month), int(day))
                    else:
                        day, month, year = match.groups()
                        return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        # Search in parent elements or siblings
        parent = element.parent
        if parent:
            parent_text = parent.get_text(strip=True)
            for pattern in date_patterns:
                match = re.search(pattern, parent_text, re.IGNORECASE)
                if match:
                    try:
                        if 'tháng' in pattern or 'th' in pattern:
                            day, _, month, year = match.groups()
                            return datetime(int(year), int(month), int(day))
                        elif pattern.startswith(r'(\d{4})'):
                            year, month, day = match.groups()
                            return datetime(int(year), int(month), int(day))
                        else:
                            day, month, year = match.groups()
                            return datetime(int(year), int(month), int(day))
                    except ValueError:
                        continue
        
        return None

    def download_pdf(self, url, filename):
        """Download PDF file with content validation"""
        try:
            response = requests.get(url, headers=self.headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type:
                logger.warning(f"URL {url} returned HTML instead of PDF - skipping")
                return None
            
            # Read first chunk to validate PDF header
            first_chunk = None
            content_chunks = []
            
            for chunk in response.iter_content(chunk_size=8192):
                if first_chunk is None:
                    first_chunk = chunk
                    # Check if this looks like a PDF file
                    if not chunk.startswith(b'%PDF'):
                        if chunk.startswith(b'<!DOCTYPE') or chunk.startswith(b'<html'):
                            logger.warning(f"URL {url} returned HTML content instead of PDF - skipping")
                            return None
                content_chunks.append(chunk)
            
            # If we get here, it should be a valid PDF
            filepath = os.path.join(self.download_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in content_chunks:
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filename}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            return None

    def download_company_pdfs(self, company_code):
        """Download PDFs for a specific company"""
        if company_code not in self.companies:
            logger.error(f"Company {company_code} not supported")
            return []

        logger.info(f"Downloading {company_code} PDFs...")
        company_config = self.companies[company_code]
        url = company_config['url']
        
        response = self.get_page_content(url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        downloaded_files = []
        
        if company_config['method'] == 'direct_pdf':
            pdf_links = self.find_pdf_links(soup, url, company_config['title_pattern'])
            
            # Only download the most recent PDF (first one after sorting)
            if pdf_links:
                link_info = pdf_links[0]  # Most recent PDF
                pdf_url = link_info['url']
                title = link_info['title']
                upload_date = link_info.get('upload_date')
                
                clean_title = re.sub(r'[^\w\s-]', '', title).strip()
                clean_title = re.sub(r'[-\s]+', '_', clean_title)
                filename = f"{company_code}_{clean_title}_{datetime.now().strftime('%Y%m%d')}.pdf"
                
                logger.info(f"Downloading latest PDF for {company_code}: {title}")
                if upload_date:
                    logger.info(f"Upload date: {upload_date.strftime('%Y-%m-%d')}")
                
                filepath = self.download_pdf(pdf_url, filename)
                if filepath:
                    downloaded_files.append({
                        'company': company_code,
                        'filepath': filepath,
                        'title': title,
                        'url': pdf_url,
                        'upload_date': upload_date
                    })
            else:
                logger.warning(f"No matching PDFs found for {company_code}")
        
        elif company_config['method'] == 'secondary_url' and company_code in ['GEG', 'PC1']:
            # Handle secondary URL method for GEG and PC1 - find latest URL with matching title
            potential_links = []
            for link in soup.find_all('a', href=True):
                link_text = link.get_text(strip=True)
                if re.search(company_config['title_pattern'], link_text, re.IGNORECASE):
                    href = link['href']
                    full_url = urljoin(url, href)
                    # Extract date from the link element
                    upload_date = self.extract_date_from_element(link)
                    potential_links.append({
                        'url': full_url,
                        'title': link_text,
                        'upload_date': upload_date
                    })
                    date_str = upload_date.strftime('%Y-%m-%d') if upload_date else 'Unknown'
                    logger.info(f"Found {company_code} link: {link_text} -> {full_url} (Date: {date_str})")
            
            # Sort potential links by date (most recent first)
            potential_links.sort(key=lambda x: (x['upload_date'] is None, 
                                              -(x['upload_date'].timestamp() if x['upload_date'] else 0)))
            
            # Use only the most recent URL
            if potential_links:
                latest_link = potential_links[0]
                logger.info(f"Using latest {company_code} URL: {latest_link['title']} " +
                           f"(Date: {latest_link['upload_date'].strftime('%Y-%m-%d') if latest_link['upload_date'] else 'Unknown'})")
                
                # Check this URL for PDFs
                secondary_response = self.get_page_content(latest_link['url'])
                if secondary_response:
                    secondary_soup = BeautifulSoup(secondary_response.content, 'html.parser')
                    pdf_links = self.find_pdf_links(secondary_soup, latest_link['url'], company_config['title_pattern'])
                    
                    # Download the most recent PDF from this page
                    if pdf_links:
                        pdf_link = pdf_links[0]  # Already sorted by date
                        pdf_url = pdf_link['url']
                        title = pdf_link['title'] or latest_link['title']
                        
                        clean_title = re.sub(r'[^\w\s-]', '', title).strip()
                        clean_title = re.sub(r'[-\s]+', '_', clean_title)
                        filename = f"{company_code}_{clean_title}_{datetime.now().strftime('%Y%m%d')}.pdf"
                        
                        logger.info(f"Downloading latest PDF for {company_code}: {title}")
                        filepath = self.download_pdf(pdf_url, filename)
                        if filepath:
                            downloaded_files.append({
                                'company': company_code,
                                'filepath': filepath,
                                'title': title,
                                'url': pdf_url,
                                'upload_date': pdf_link.get('upload_date')
                            })
                    else:
                        logger.warning(f"No PDFs found on {company_code} page: {latest_link['url']}")
                else:
                    logger.warning(f"Could not access {company_code} secondary URL: {latest_link['url']}")
            else:
                logger.warning(f"No matching URLs found for {company_code}")
                
                time.sleep(1)  # Be respectful with requests
        
        return downloaded_files

    def download_all_companies(self):
        """Download PDFs for all companies"""
        all_downloads = []
        
        for company_code in self.companies.keys():
            try:
                logger.info(f"Processing {company_code}...")
                downloads = self.download_company_pdfs(company_code)
                all_downloads.extend(downloads)
                logger.info(f"Downloaded {len(downloads)} files for {company_code}")
                time.sleep(2)  # Be respectful between companies
            except Exception as e:
                logger.error(f"Error downloading {company_code} PDFs: {str(e)}")
        
        return all_downloads

    def display_processing_guide(self):
        """Display the processing requirements for each company"""
        logger.info("=== COMPANY PROCESSING GUIDE ===")
        for company_code, config in self.companies.items():
            logger.info(f"\n{company_code}:")
            logger.info(f"  - Title Pattern: {config['title_pattern']}")
            logger.info(f"  - Method: {config['method']}")
            logger.info(f"  - Data Source: {config['data_source']}")
            logger.info(f"  - CSV File: {config['csv_file']}")
            
            if company_code == 'REE':
                logger.info("  - Specific: Download PDF files titled 'ree_date_results...' with latest date")
                logger.info("  - Data: Read from Năng Lượng pages")
            elif company_code == 'HDG':
                logger.info("  - Specific: Download PDF files titled 'Ban Tin IR Date' with latest date")
                logger.info("  - Data: Năng Lượng page (Solar: Hồng Phong 4, Infra 1; Wind: 7A; Rest: Hydro)")
            elif company_code == 'GEG':
                logger.info("  - Specific: Find secondary URL with 'kết quả kinh doanh...' content, latest date")
                logger.info("  - Data: Read from URL content")
            elif company_code == 'PC1':
                logger.info("  - Specific: Find secondary URL with 'bản tin nhà đầu tư...' title, latest date")
                logger.info("  - Data: Read from URL content")
        logger.info("===============================\n")

# ================================================================================================
# PDF PROCESSOR CLASS
# ================================================================================================

class CompanyPDFProcessor:
    def __init__(self, openai_api_key: str = None):
        """Initialize the PDF processor with OpenAI client"""
        self.api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. OpenAI features will be disabled.")
            logger.warning("Please set OPENAI_API_KEY environment variable to enable AI data extraction.")
        else:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                logger.warning("OpenAI features will be disabled.")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def load_pdf(self, pdf_path: str) -> List[str]:
        """Load PDF and extract text using LangChain PyPDFLoader"""
        try:
            logger.info(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Extract text from all pages
            full_text = "\n".join([doc.page_content for doc in documents])
            
            # Split into manageable chunks
            chunks = self.text_splitter.split_text(full_text)
            
            logger.info(f"Extracted {len(chunks)} text chunks from {len(documents)} pages")
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            return []
    
    def identify_report_period(self, text: str) -> Dict[str, Any]:
        """Identify the reporting period from Vietnamese text"""
        
        # Vietnamese quarter patterns
        period_patterns = {
            # Quarterly patterns
            r'quý\s+I\s+(\d{4})': {'type': 'quarterly', 'quarter': '1Q'},
            r'quý\s+1\s+(\d{4})': {'type': 'quarterly', 'quarter': '1Q'},
            r'3\s+tháng\s+đầu\s+năm\s+(\d{4})': {'type': 'quarterly', 'quarter': '1Q'},
            
            r'quý\s+II\s+(\d{4})': {'type': 'quarterly', 'quarter': '2Q'},
            r'quý\s+2\s+(\d{4})': {'type': 'quarterly', 'quarter': '2Q'},
            r'6\s+tháng\s+(\d{4})': {'type': 'cumulative', 'quarter': '2Q', 'cumulative_quarters': 2},
            
            r'quý\s+III\s+(\d{4})': {'type': 'quarterly', 'quarter': '3Q'},
            r'quý\s+3\s+(\d{4})': {'type': 'quarterly', 'quarter': '3Q'},
            r'9\s+tháng\s+(\d{4})': {'type': 'cumulative', 'quarter': '3Q', 'cumulative_quarters': 3},
            
            r'quý\s+IV\s+(\d{4})': {'type': 'quarterly', 'quarter': '4Q'},
            r'quý\s+4\s+(\d{4})': {'type': 'quarterly', 'quarter': '4Q'},
            r'12\s+tháng\s+(\d{4})': {'type': 'cumulative', 'quarter': '4Q', 'cumulative_quarters': 4},
            r'cả\s+năm\s+(\d{4})': {'type': 'cumulative', 'quarter': '4Q', 'cumulative_quarters': 4},
            r'năm\s+(\d{4})': {'type': 'cumulative', 'quarter': '4Q', 'cumulative_quarters': 4},
        }
        
        for pattern, info in period_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                return {
                    'year': year,
                    'quarter': info['quarter'],
                    'type': info['type'],
                    'cumulative_quarters': info.get('cumulative_quarters', 1),
                    'raw_text': match.group(0)
                }
        
        # Default to current year 4Q if no pattern found
        current_year = datetime.now().year
        return {
            'year': current_year,
            'quarter': '4Q',
            'type': 'quarterly',
            'cumulative_quarters': 1,
            'raw_text': 'Unknown period'
        }
    
    def extract_financial_data_with_openai(self, text_chunks: List[str], company_code: str) -> QuarterlyFinancialData:
        """Use OpenAI with structured outputs to extract financial data"""
        
        # Check if OpenAI client is available
        if not self.client:
            logger.warning("OpenAI client not available. Cannot extract financial data with AI.")
            logger.info("Returning empty data structure. Please set OPENAI_API_KEY to enable AI extraction.")
            current_year = datetime.now().year
            return QuarterlyFinancialData(
                company_code=company_code,
                year=current_year,
                quarter='4Q',
                report_period='Unknown - OpenAI unavailable',
                extraction_confidence=0.0
            )
        
        # Combine text chunks for analysis
        combined_text = "\n".join(text_chunks[:3])  # Use first 3 chunks to avoid token limits
        
        # Identify reporting period
        period_info = self.identify_report_period(combined_text)
        
        system_prompt = f"""
        You are a financial data extraction expert specializing in Vietnamese corporate financial reports.
        
        Extract financial data from the provided Vietnamese financial report text for company {company_code}.
        
        Key Instructions:
        1. Look for financial figures in Vietnamese (doanh thu, lợi nhuận, tài sản, etc.)
        2. Convert all monetary values to billion VND (if in million, divide by 1000)
        3. Handle different Vietnamese number formats (1.234,56 or 1,234.56)
        4. If data appears to be cumulative (6 tháng, 9 tháng, cả năm), note this for later adjustment
        5. Look for power-related metrics if this is a power company (generation, capacity)
        6. Set extraction_confidence based on how clear the data is (0.0 to 1.0)
        
        Vietnamese Financial Terms:
        - Doanh thu = Revenue
        - Lợi nhuận gộp = Gross Profit
        - Lợi nhuận từ hoạt động kinh doanh = Operating Profit
        - Lợi nhuận sau thuế = Net Profit After Tax
        - EBITDA = EBITDA
        - Tổng tài sản = Total Assets
        - Vốn chủ sở hữu = Total Equity
        - Tiền và tương đương tiền = Cash and Equivalents
        
        Report Period Detected: {period_info['quarter']} {period_info['year']} ({period_info['type']})
        """
        
        user_prompt = f"""
        Company: {company_code}
        Report Period: {period_info['raw_text']}
        
        Financial Report Text:
        {combined_text[:8000]}  # Limit to avoid token limits
        
        Extract the financial data according to the structured format.
        """
        
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",  # Use the latest model with structured outputs
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=QuarterlyFinancialData,
                temperature=0.1
            )
            
            financial_data = response.choices[0].message.parsed
            
            # Update with detected period info
            financial_data.company_code = company_code
            financial_data.year = period_info['year']
            financial_data.quarter = period_info['quarter']
            financial_data.report_period = period_info['raw_text']
            
            return financial_data
            
        except Exception as e:
            logger.error(f"Error extracting financial data with OpenAI: {str(e)}")
            
            # Return empty data structure with detected period
            return QuarterlyFinancialData(
                company_code=company_code,
                year=period_info['year'],
                quarter=period_info['quarter'],
                report_period=period_info['raw_text'],
                extraction_confidence=0.0
            )
    
    def process_pdf(self, pdf_path: str, company_code: str) -> QuarterlyFinancialData:
        """Process a single PDF and extract financial data"""
        logger.info(f"Processing PDF for {company_code}: {pdf_path}")
        
        # Load and extract text
        text_chunks = self.load_pdf(pdf_path)
        if not text_chunks:
            logger.error(f"No text extracted from {pdf_path}")
            return None
        
        # Extract financial data using OpenAI
        financial_data = self.extract_financial_data_with_openai(text_chunks, company_code)
        
        logger.info(f"Extracted data for {company_code} {financial_data.quarter} {financial_data.year}")
        logger.info(f"Revenue: {financial_data.revenue}, Net Profit: {financial_data.net_profit}")
        
        return financial_data

# ================================================================================================
# CSV UPDATER CLASS
# ================================================================================================

class CompanyDataUpdater:
    def __init__(self, data_dir: str = "../data"):
        """Initialize the CSV updater"""
        self.data_dir = data_dir
        
        # Mapping of company codes to CSV files
        self.company_files = {
            'REE': 'company_ree_monthly.csv',
            'PC1': 'company_pc1_monthly.csv', 
            'HDG': 'company_hdg_monthly.csv',
            'GEG': 'company_geg_monthly.csv',
            'POW': 'company_pow_monthly.csv'  # Will be created if needed
        }
        
        # Quarter to month mapping for quarterly data
        self.quarter_months = {
            '1Q': 3,   # Q1 ends in March
            '2Q': 6,   # Q2 ends in June  
            '3Q': 9,   # Q3 ends in September
            '4Q': 12   # Q4 ends in December
        }
    
    def load_existing_data(self, company_code: str) -> Optional[pd.DataFrame]:
        """Load existing CSV data for a company"""
        filename = self.company_files.get(company_code)
        if not filename:
            logger.warning(f"No CSV file configured for company {company_code}")
            return None
        
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"CSV file does not exist: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath)
            
            # Ensure Date column exists and is datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.rename(columns={'date': 'Date'})
            else:
                logger.error(f"No Date column found in {filepath}")
                return None
            
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            return None
    
    def create_date_from_quarter(self, year: int, quarter: str) -> datetime:
        """Create a date from year and quarter"""
        month = self.quarter_months.get(quarter, 12)
        return datetime(year, month, 1)  # Use 1st day of the ending month
    
    def map_financial_data_to_columns(self, financial_data: QuarterlyFinancialData, 
                                    existing_columns: List[str]) -> Dict[str, float]:
        """Map financial data to existing CSV columns"""
        
        # Power industry specific column mappings based on actual CSV structures
        column_mappings = {
            # Revenue mappings - match actual CSV column names
            'revenue': ['revenue', 'doanh_thu', 'net_revenue', 'total_revenue', 
                       'solar revenue', 'wind revenue', 'hydro revenue', 'others revenue'],
            'gross_profit': ['gross_profit', 'loi_nhuan_gop', 'gross_margin'],
            'operating_profit': ['operating_profit', 'operating_income', 'ebit', 'loi_nhuan_hoat_dong'],
            'net_profit': ['net_profit', 'net_income', 'loi_nhuan_sau_thue', 'profit_after_tax',
                          'solar npat', 'wind npat', 'hydro npat', 'thermal & others npat'],
            'ebitda': ['ebitda'],
            
            # Balance sheet mappings
            'total_assets': ['total_assets', 'tong_tai_san', 'assets'],
            'total_equity': ['total_equity', 'von_chu_so_huu', 'equity'],
            'cash_and_equivalents': ['cash', 'cash_equivalents', 'tien_mat'],
            
            # Power-specific mappings - match actual CSV structures
            'power_generation_gwh': ['generation', 'power_generation', 'electricity_generation',
                                   'solar volume', 'wind volume', 'hydro volume', 'thermal volume'],
            'capacity_mw': ['capacity', 'installed_capacity', 'power_capacity']
        }
        
        # Create a mapping of actual columns to values
        column_values = {}
        existing_columns_lower = [col.lower() for col in existing_columns]
        
        for field_name, possible_columns in column_mappings.items():
            field_value = getattr(financial_data, field_name)
            
            if field_value is not None:
                # Find matching column in existing CSV
                for possible_col in possible_columns:
                    for i, existing_col in enumerate(existing_columns_lower):
                        if possible_col in existing_col or existing_col in possible_col:
                            actual_column = existing_columns[i]
                            column_values[actual_column] = field_value
                            logger.info(f"Mapped {field_name} ({field_value}) to column '{actual_column}'")
                            break
                    if actual_column:
                        break
        
        return column_values
    
    def update_csv_with_quarterly_data(self, financial_data: QuarterlyFinancialData) -> bool:
        """Update CSV file with quarterly financial data"""
        
        company_code = financial_data.company_code
        logger.info(f"Updating CSV for {company_code} - {financial_data.quarter} {financial_data.year}")
        
        # Load existing data
        df = self.load_existing_data(company_code)
        if df is None:
            logger.error(f"Cannot load existing data for {company_code}")
            return False
        
        # Create date for this quarter
        quarter_date = self.create_date_from_quarter(financial_data.year, financial_data.quarter)
        
        # Check if this quarter already exists
        existing_row = df[df['Date'] == quarter_date]
        
        # Map financial data to CSV columns
        column_values = self.map_financial_data_to_columns(financial_data, df.columns.tolist())
        
        if column_values:
            if len(existing_row) > 0:
                # Update existing row
                row_index = existing_row.index[0]
                for column, value in column_values.items():
                    df.loc[row_index, column] = value
                logger.info(f"Updated existing row for {quarter_date}")
            else:
                # Create new row
                new_row = {'Date': quarter_date}
                new_row.update(column_values)
                
                # Add the new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                logger.info(f"Added new row for {quarter_date}")
            
            # Sort by date
            df = df.sort_values('Date')
            
            # Save updated CSV
            filename = self.company_files[company_code]
            filepath = os.path.join(self.data_dir, filename)
            
            try:
                df.to_csv(filepath, index=False)
                logger.info(f"Successfully updated {filepath}")
                return True
            except Exception as e:
                logger.error(f"Error saving {filepath}: {str(e)}")
                return False
        else:
            logger.warning(f"No matching columns found for {company_code} data")
            return False

# ================================================================================================
# MAIN PIPELINE CLASS
# ================================================================================================

class FinancialDataPipeline:
    def __init__(self, openai_api_key: str = None, data_dir: str = "../data"):
        """Initialize the complete pipeline"""
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        # Initialize components
        self.downloader = CompanyPDFDownloader()
        self.processor = CompanyPDFProcessor(self.openai_api_key)
        self.updater = CompanyDataUpdater(data_dir)
        
        # Results storage
        self.results = {
            'downloads': [],
            'extractions': [],
            'updates': {},
            'errors': []
        }
    
    def run_full_pipeline(self, companies: List[str] = None) -> Dict[str, Any]:
        """Run the complete pipeline for specified companies"""
        
        if companies is None:
            companies = ['REE', 'PC1', 'HDG', 'GEG', 'POW']
        
        # Display processing guide
        self.downloader.display_processing_guide()
        
        logger.info(f"Starting financial data pipeline for companies: {companies}")
        
        try:
            # Step 1: Download PDFs
            logger.info("Step 1: Downloading PDFs...")
            downloaded_files = self.download_pdfs(companies)
            self.results['downloads'] = downloaded_files
            
            if not downloaded_files:
                logger.error("No PDFs downloaded. Pipeline stopped.")
                return self.results
            
            # Step 2: Process PDFs and extract financial data
            logger.info("Step 2: Processing PDFs and extracting financial data...")
            financial_data = self.process_pdfs(downloaded_files)
            self.results['extractions'] = financial_data
            
            if not financial_data:
                logger.error("No financial data extracted. Pipeline stopped.")
                return self.results
            
            # Step 3: Update CSV files
            logger.info("Step 3: Updating CSV files...")
            update_results = self.update_csv_files(financial_data)
            self.results['updates'] = update_results
            
            # Step 4: Generate summary
            logger.info("Step 4: Generating summary...")
            summary = self.generate_pipeline_summary()
            
            logger.info("Pipeline completed successfully!")
            logger.info(summary)
            
            return self.results
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return self.results
    
    def download_pdfs(self, companies: List[str]) -> List[Dict]:
        """Download PDFs for specified companies"""
        try:
            all_downloads = []
            
            for company in companies:
                downloads = self.downloader.download_company_pdfs(company)
                all_downloads.extend(downloads)
            
            logger.info(f"Downloaded {len(all_downloads)} PDFs for {len(companies)} companies")
            return all_downloads
            
        except Exception as e:
            error_msg = f"Error downloading PDFs: {str(e)}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return []
    
    def process_pdfs(self, downloaded_files: List[Dict]) -> List[QuarterlyFinancialData]:
        """Process downloaded PDFs and extract financial data"""
        financial_data = []
        
        for pdf_info in downloaded_files:
            try:
                logger.info(f"Processing {pdf_info['company']}: {pdf_info['title']}")
                
                data = self.processor.process_pdf(pdf_info['filepath'], pdf_info['company'])
                if data and data.extraction_confidence > 0.3:  # Only keep high-confidence extractions
                    financial_data.append(data)
                    logger.info(f"Successfully extracted data for {data.company_code} {data.quarter} {data.year}")
                else:
                    logger.warning(f"Low confidence extraction for {pdf_info['company']}")
                    
            except Exception as e:
                error_msg = f"Error processing {pdf_info['filepath']}: {str(e)}"
                logger.error(error_msg)
                self.results['errors'].append(error_msg)
        
        logger.info(f"Successfully processed {len(financial_data)} PDFs")
        return financial_data
    
    def update_csv_files(self, financial_data: List[QuarterlyFinancialData]) -> Dict[str, bool]:
        """Update CSV files with financial data"""
        try:
            update_results = {}
            
            for data in financial_data:
                success = self.updater.update_csv_with_quarterly_data(data)
                update_results[data.company_code] = success
            
            successful_updates = sum(1 for success in update_results.values() if success)
            total_companies = len(update_results)
            
            logger.info(f"CSV update results: {successful_updates}/{total_companies} companies updated successfully")
            
            return update_results
            
        except Exception as e:
            error_msg = f"Error updating CSV files: {str(e)}"
            logger.error(error_msg)
            self.results['errors'].append(error_msg)
            return {}
    
    def generate_pipeline_summary(self) -> str:
        """Generate a comprehensive summary of the pipeline execution"""
        
        summary = "\n" + "="*80 + "\n"
        summary += "📊 FINANCIAL DATA PIPELINE SUMMARY\n"
        summary += "="*80 + "\n\n"
        
        # Download summary
        downloads = self.results.get('downloads', [])
        summary += f"📥 DOWNLOAD PHASE:\n"
        summary += f"   Total PDFs Downloaded: {len(downloads)}\n"
        
        download_by_company = {}
        for download in downloads:
            company = download['company']
            download_by_company[company] = download_by_company.get(company, 0) + 1
        
        for company, count in download_by_company.items():
            summary += f"   - {company}: {count} files\n"
        
        # Extraction summary
        extractions = self.results.get('extractions', [])
        summary += f"\n🔍 EXTRACTION PHASE:\n"
        summary += f"   Total Records Extracted: {len(extractions)}\n"
        
        extraction_by_company = {}
        for extraction in extractions:
            company = extraction.company_code
            extraction_by_company[company] = extraction_by_company.get(company, 0) + 1
        
        for company, count in extraction_by_company.items():
            summary += f"   - {company}: {count} records\n"
        
        # Update summary
        updates = self.results.get('updates', {})
        summary += f"\n💾 UPDATE PHASE:\n"
        
        successful_updates = sum(1 for success in updates.values() if success)
        total_companies = len(updates)
        
        summary += f"   Companies Updated: {successful_updates}/{total_companies}\n"
        
        for company, success in updates.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            summary += f"   - {company}: {status}\n"
        
        # Error summary
        errors = self.results.get('errors', [])
        if errors:
            summary += f"\n⚠️ ERRORS:\n"
            for i, error in enumerate(errors, 1):
                summary += f"   {i}. {error}\n"
        else:
            summary += f"\n✅ NO ERRORS ENCOUNTERED\n"
        
        # Overall status
        overall_success = len(errors) == 0 and successful_updates == total_companies
        status_emoji = "🎉" if overall_success else "⚠️"
        
        summary += f"\n{status_emoji} OVERALL STATUS: "
        summary += "SUCCESSFUL" if overall_success else "COMPLETED WITH ISSUES"
        summary += f"\n\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += "="*80 + "\n"
        
        return summary

# ================================================================================================
# MAIN EXECUTION FUNCTIONS
# ================================================================================================

def main():
    """Main function - can be used for testing or running the pipeline"""
    print("🏢 Vietnamese Power Company Financial Data Pipeline")
    print("=" * 60)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Please set the OPENAI_API_KEY environment variable")
        return
    
    try:
        # Initialize pipeline
        pipeline = FinancialDataPipeline(api_key)
        
        # Ask user which companies to process
        print("\nAvailable companies: REE, PC1, HDG, GEG, POW")
        user_input = input("Enter companies to process (comma-separated, or press Enter for all): ").strip()
        
        if user_input:
            companies = [company.strip().upper() for company in user_input.split(',')]
        else:
            companies = ['REE', 'PC1', 'HDG', 'GEG', 'POW']
        
        print(f"\n🚀 Starting pipeline for companies: {companies}")
        
        # Run the pipeline
        results = pipeline.run_full_pipeline(companies)
        
        print("\n✅ Pipeline completed! Check the logs above for detailed results.")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")


def test_components():
    """Test individual components"""
    print("🧪 Testing Individual Components")
    print("=" * 40)
    
    # Test 1: PDF Downloader
    print("\n1. Testing PDF Downloader...")
    downloader = CompanyPDFDownloader()
    downloads = downloader.download_company_pdfs('REE')
    print(f"   Downloaded {len(downloads)} files for REE")
    
    # Test 2: PDF Processor (if OpenAI key available)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and downloads:
        print("\n2. Testing PDF Processor...")
        processor = CompanyPDFProcessor(api_key)
        
        # Process first downloaded file
        first_pdf = downloads[0]
        result = processor.process_pdf(first_pdf['filepath'], first_pdf['company'])
        
        if result:
            print(f"   Extracted: {result.company_code} {result.quarter} {result.year}")
            print(f"   Revenue: {result.revenue} billion VND")
            print(f"   Confidence: {result.extraction_confidence}")
    
    print("\n✅ Component testing completed!")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run full pipeline")
    print("2. Test components")
    
    choice = input("Enter choice (1-2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_components()
    else:
        print("Invalid choice. Running full pipeline...")
        main()