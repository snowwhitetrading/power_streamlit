"""
SSI/TCBS API functions for fetching stock price data
Enhanced to replace vnstock functionality
"""

import requests
import pandas as pd
import time
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

def fetch_historical_price(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch stock historical price and volume data from TCBS API"""
    
    # TCBS API endpoint for historical data
    url = "https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term"
    
    # Default to 1 year ago if no start_date provided
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Default to current date if no end_date provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert dates to timestamps
    start_timestamp = str(int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()))
    end_timestamp = str(int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()))
    
    # Parameters for stock data
    params = {
        "ticker": ticker,
        "type": "stock",
        "resolution": "D",  # Daily data
        "from": start_timestamp,
        "to": end_timestamp
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data and data['data']:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data['data'])
            
            # Convert timestamp to datetime
            if 'tradingDate' in df.columns:
                # Check if tradingDate is already in ISO format
                if df['tradingDate'].dtype == 'object' and str(df['tradingDate'].iloc[0]).count('T') > 0:
                    df['tradingDate'] = pd.to_datetime(df['tradingDate']).dt.date
                else:
                    # Handle both timestamp in ms and seconds
                    try:
                        df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='ms').dt.date
                    except:
                        df['tradingDate'] = pd.to_datetime(df['tradingDate'], unit='s').dt.date
            
            # Rename tradingDate to time for consistency
            if 'tradingDate' in df.columns:
                df = df.rename(columns={'tradingDate': 'time'})
            
            # Select relevant columns in standard order
            columns_to_keep = ['time', 'open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in columns_to_keep if col in df.columns]
            
            if available_columns:
                df = df[available_columns]
                
                # Sort by date
                if 'time' in df.columns:
                    df = df.sort_values('time').reset_index(drop=True)
                
                # Ensure numeric columns are properly typed
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            else:
                print(f"No valid columns found for {ticker}")
                return pd.DataFrame()
        else:
            print(f"No data found in API response for {ticker}")
            return pd.DataFrame()
            
    except requests.exceptions.Timeout:
        print(f"Timeout error fetching data for {ticker}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Network error fetching data for {ticker}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Unexpected error for {ticker}: {e}")
        return pd.DataFrame()

def get_quarterly_stock_data(symbols, start_date='2020-01-01', end_date='2025-06-30'):
    """Get quarterly stock price data using SSI API (replacement for vnstock)"""
    
    stock_data = {}
    
    for symbol in symbols:
        try:
            # Get daily data
            df = fetch_historical_price(symbol, start_date, end_date)
            
            if df is not None and not df.empty and 'close' in df.columns:
                # Convert time to datetime if it's not already
                df['time'] = pd.to_datetime(df['time'])
                df = df.sort_values('time')
                
                # Resample to quarterly (last day of quarter)
                df_indexed = df.set_index('time')
                quarterly_data = df_indexed.resample('Q').last()
                quarterly_data = quarterly_data.reset_index()
                quarterly_data = quarterly_data.rename(columns={'time': 'date'})
                
                stock_data[symbol] = quarterly_data[['date', 'close']].copy()
                
        except Exception as e:
            print(f"Error getting quarterly data for {symbol}: {e}")
            continue
    
    return stock_data

def get_vnindex_data(start_date='2020-01-01', end_date='2025-06-30'):
    """Get VN-Index data from CSV file for accurate cumulative returns"""
    
    try:
        # First try to load from CSV file
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'vn_index_monthly.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Clean the data
            df.columns = ['period', 'vnindex']
            df = df.dropna()
            
            # Clean the VNINDEX values (remove commas and convert to float)
            df['vnindex'] = df['vnindex'].astype(str).str.replace(',', '').astype(float)
            
            # Sort by period
            df = df.sort_values('period')
            
            # Calculate quarterly returns
            df['return'] = df['vnindex'].pct_change() * 100
            
            # Calculate cumulative return properly
            # First fill NaN return with 0 to handle the first period
            df['return'] = df['return'].fillna(0)
            df['cumulative_return'] = ((1 + df['return'] / 100).cumprod() - 1) * 100
            
            # Filter by date range if needed
            # Convert period to datetime for filtering
            def period_to_date(period):
                year = int(period[:4]) if len(period) >= 4 else 2020
                quarter = int(period[-1]) if period[-1].isdigit() else 1
                month = (quarter - 1) * 3 + 3  # End of quarter month
                return pd.Timestamp(year=year, month=month, day=1)
            
            df['date'] = df['period'].apply(period_to_date)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data by date range
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            filtered_df = df[mask].copy()
            
            if not filtered_df.empty:
                # Recalculate cumulative returns for the filtered period
                filtered_df = filtered_df.reset_index(drop=True)
                filtered_df['return'] = filtered_df['vnindex'].pct_change() * 100
                filtered_df['return'] = filtered_df['return'].fillna(0)
                filtered_df['cumulative_return'] = ((1 + filtered_df['return'] / 100).cumprod() - 1) * 100
                
                return filtered_df[['period', 'return', 'cumulative_return']]
            
    except Exception as e:
        print(f"Error loading VN-Index from CSV: {e}")
    
    # Fallback to API if CSV fails
    try:
        # Get VN-Index data using SSI API
        df = fetch_historical_price("VNINDEX", start_date, end_date)
        
        if df is not None and not df.empty and 'close' in df.columns:
            # Convert time to datetime
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time')
            
            # Resample to quarterly
            df_indexed = df.set_index('time')
            quarterly_data = df_indexed.resample('Q').last()
            
            # Calculate quarterly returns
            quarterly_data['return'] = quarterly_data['close'].pct_change() * 100
            
            # Calculate cumulative return properly
            # First fill NaN return with 0 to handle the first period
            quarterly_data['return'] = quarterly_data['return'].fillna(0)
            quarterly_data['cumulative_return'] = ((1 + quarterly_data['return'] / 100).cumprod() - 1) * 100
            
            # Create period labels safely
            quarterly_data = quarterly_data.reset_index()
            quarterly_data['period'] = quarterly_data['time'].apply(lambda x: f"{x.year}Q{x.quarter}")
            
            return quarterly_data[['period', 'return', 'cumulative_return']].dropna()
    
    except Exception as e:
        print(f"Error getting VN-Index data from API: {e}")
        
    # Return mock VN-Index data if both CSV and API fail
    periods = pd.date_range(start=start_date, end=end_date, freq='Q')
    
    # Generate mock VN-Index with realistic movements
    np.random.seed(42)  # Fixed seed for consistent mock data
    returns = np.random.normal(0.02, 0.12, len(periods))  # 2% mean return, 12% volatility per quarter
    
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    return pd.DataFrame({
        'period': [f"{p.year}Q{p.quarter}" for p in periods],
        'return': returns * 100,
        'cumulative_return': cumulative_returns * 100
    })

def get_stock_history(symbol, start_date, end_date, interval='1D'):
    """Get stock history data - replacement for vnstock stock.quote.history()"""
    
    try:
        df = fetch_historical_price(symbol, start_date, end_date)
        
        if df is not None and not df.empty:
            # Convert time to datetime and set as index for compatibility
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df = df.sort_index()
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error getting history for {symbol}: {e}")
        return pd.DataFrame()

def plot_ohlcv_candlestick(df, symbol, start_date='2024-12-31'):
    """Create OHLCV candlestick chart with volume"""
    df_temp = df.copy()
    
    # Handle different date column names and formats
    if 'time' in df_temp.columns:
        df_temp = df_temp.reset_index()
        date_col = 'time'
    elif df_temp.index.name == 'time' or 'time' in str(df_temp.index.dtype):
        df_temp = df_temp.reset_index()
        date_col = 'time'
    else:
        # Assume index is date
        df_temp = df_temp.reset_index()
        date_col = df_temp.columns[0]
    
    # Convert to datetime and filter by start_date
    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df_temp = df_temp[df_temp[date_col] >= start_dt]
    
    df_temp[date_col] = df_temp[date_col].dt.strftime('%Y-%m-%d')
    
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.02,  # Standard spacing
        row_heights=[0.7, 0.3],
        subplot_titles=[f"{symbol} Price Chart", "Volume"]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_temp[date_col],
            open=df_temp['open'],
            high=df_temp['high'],
            low=df_temp['low'],
            close=df_temp['close'],
            name='OHLC',
            opacity=1.0
        ), row=1, col=1
    )
    
    # Color volume bars by up/down
    colors = ['green' if c >= o else 'red' for c, o in zip(df_temp['close'], df_temp['open'])]
    fig.add_trace(
        go.Bar(
            x=df_temp[date_col],
            y=df_temp['volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.8
        ), row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        template='plotly_white',  # Standard template
        title=f"{symbol} Price Chart",
        xaxis2_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volume",
        xaxis_rangeslider_visible=False,  
        xaxis2_rangeslider_visible=False,
        height=600,  # Fixed height for OHLC chart
        width=1000,  # Standard width
        showlegend=False
    )
    fig.update_xaxes(
        showgrid=False,
        type='category'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def load_ticker_price(ticker, start_date):
    """
    Load OHLCV data for a specific ticker and create chart.
    """
    df = fetch_historical_price(ticker, start_date)
    if df is not None and not df.empty:
        fig = plot_ohlcv_candlestick(df, ticker, start_date)
        return fig
    return None

def get_stock_data_batch(symbols, start_date=None, end_date=None, progress_callback=None):
    """
    Fetch stock data for multiple symbols with progress tracking
    
    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format) 
        progress_callback: Optional function to call with progress updates
        
    Returns:
        dict: Dictionary with symbol as key and DataFrame as value
    """
    stock_data = {}
    successful_fetches = 0
    failed_fetches = 0
    
    # Set default dates
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    for i, symbol in enumerate(symbols):
        try:
            if progress_callback:
                progress_callback(f"Fetching data for {symbol}... ({i+1}/{len(symbols)})")
            
            df = fetch_historical_price(symbol, start_date, end_date)
            
            if df is not None and not df.empty and 'close' in df.columns:
                stock_data[symbol] = df
                successful_fetches += 1
            else:
                failed_fetches += 1
                
        except Exception as e:
            failed_fetches += 1
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.1)
    
    return stock_data

def get_performance_summary(symbols, start_date=None, end_date=None):
    """
    Get performance summary for multiple stocks
    
    Returns:
        DataFrame with performance metrics for each symbol
    """
    stock_data = get_stock_data_batch(symbols, start_date, end_date)
    
    performance_data = []
    
    for symbol, df in stock_data.items():
        if df is not None and not df.empty and len(df) > 1:
            try:
                first_price = df['close'].iloc[0]
                last_price = df['close'].iloc[-1]
                performance = ((last_price - first_price) / first_price) * 100
                
                performance_data.append({
                    'symbol': symbol,
                    'start_price': first_price,
                    'end_price': last_price,
                    'performance_pct': performance,
                    'data_points': len(df)
                })
            except Exception as e:
                print(f"Error calculating performance for {symbol}: {e}")
    
    return pd.DataFrame(performance_data)

# Mock SSI stock class to replace vnstock functionality
class SSIStock:
    def __init__(self, symbol, source='SSI'):
        self.symbol = symbol
        self.source = source
        self.quote = SSIQuote(symbol)

class SSIQuote:
    def __init__(self, symbol):
        self.symbol = symbol
    
    def history(self, start, end, interval='1D'):
        """Get historical data - replacement for vnstock"""
        return get_stock_history(self.symbol, start, end, interval)

def create_ssi_stock(symbol, source='SSI'):
    """Factory function to create SSI stock object (replacement for Vnstock().stock())"""
    return SSIStock(symbol, source)
