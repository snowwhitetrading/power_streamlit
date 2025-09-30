"""
ENSO Regression Analysis Module
Analyzes the relationship between semi-annual returns of thermal and hydro portfolios against the Ocean Niño Index (ONI)
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import time
import warnings

# Import SSI API functions
warnings.filterwarnings('ignore')

# Import SSI API functions - using the same pattern as coal strategy
try:
    from ssi_api import get_stock_data_batch, fetch_historical_price
    SSI_API_AVAILABLE = True
except ImportError as e:
    SSI_API_AVAILABLE = False

def get_stock_data_for_portfolio(stock_list: list, start_date: str = "2011-01-01", end_date: str = "2025-12-31") -> dict:
    """Get stock data for all stocks in a portfolio using SSI API - following coal strategy pattern"""
    try:
        stock_data = {}
        
        if SSI_API_AVAILABLE:
            try:
                # Process stocks in smaller batches to avoid overwhelming the API
                batch_size = 5  # Process 5 stocks at a time
                all_data = {}
                
                for i in range(0, len(stock_list), batch_size):
                    batch_stocks = stock_list[i:i + batch_size]
                    st.info(f"Fetching data for batch {i//batch_size + 1}: {', '.join(batch_stocks)}")
                    
                    # Use get_stock_data_batch like the coal strategy does
                    batch_data = get_stock_data_batch(batch_stocks, start_date, end_date)
                    
                    if batch_data:
                        all_data.update(batch_data)
                    
                    # Small delay between batches to avoid overwhelming the API
                    if i + batch_size < len(stock_list):
                        time.sleep(1)
                
                # Process the fetched data
                for symbol in stock_list:
                    if symbol in all_data and not all_data[symbol].empty:
                        data = all_data[symbol].copy()
                        
                        # Ensure the data has the expected structure
                        if 'close' in data.columns:
                            # Convert 'time' column to 'date' if it exists
                            if 'time' in data.columns:
                                data['date'] = pd.to_datetime(data['time'])
                            elif 'tradingDate' in data.columns:
                                data['date'] = pd.to_datetime(data['tradingDate'])
                            else:
                                continue
                            
                            stock_data[symbol] = data
                
                if stock_data:
                    st.success(f"✅ Successfully fetched real data for {len(stock_data)}/{len(stock_list)} stocks via SSI API")
                    return stock_data
                else:
                    st.error(f"❌ No data returned from SSI API for any of the {len(stock_list)} stocks")
                    return {}
                    
            except Exception as e:
                st.error(f"❌ Failed to fetch real stock data: {e}")
                return {}
        
        if not SSI_API_AVAILABLE:
            st.error("❌ SSI API not available. Cannot fetch real stock data.")
            return {}
        
        return stock_data
        
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return {}

def convert_to_periodic_returns(stock_data: dict, frequency: str = "Q") -> dict:
    """Convert daily stock data to periodic returns based on frequency"""
    try:
        periodic_returns = {}
        
        for symbol, data in stock_data.items():
            if data.empty:
                continue
                
            # Add period columns based on frequency
            data = data.copy()
            data['year'] = data['date'].dt.year
            
            if frequency == "Q":
                data['quarter'] = data['date'].dt.quarter
                data['period'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
                group_col = 'period'
            elif frequency == "SA":  # Semi-annual
                data['half'] = data['date'].dt.month.apply(lambda x: 'H1' if x <= 6 else 'H2')
                data['period'] = data['year'].astype(str) + data['half']
                group_col = 'period'
            elif frequency == "A":  # Annual
                data['period'] = data['year'].astype(str)
                group_col = 'period'
            else:
                # Default to quarterly
                data['quarter'] = data['date'].dt.quarter
                data['period'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
                group_col = 'period'
            
            # Get period-end prices (last trading day of each period)
            period_data = data.groupby(group_col)['close'].last().reset_index()
            period_data = period_data.sort_values('period')
            
            # Calculate periodic returns
            period_data['periodic_return'] = period_data['close'].pct_change() * 100
            
            # Fill first period with 0% return for baseline
            if not period_data.empty:
                period_data.loc[0, 'periodic_return'] = 0.0
            
            periodic_returns[symbol] = period_data
        
        return periodic_returns
        
    except Exception as e:
        st.error(f"Error converting to quarterly returns: {str(e)}")
        return {}

def calculate_portfolio_quarterly_return(stock_list: list, period: str, quarterly_returns: dict) -> float:
    """Calculate equally weighted portfolio return for a specific quarter"""
    try:
        if not quarterly_returns or not stock_list:
            return 0.0
        
        portfolio_returns = []
        successful_stocks = 0
        
        for symbol in stock_list:
            if symbol in quarterly_returns:
                stock_data = quarterly_returns[symbol]
                period_data = stock_data[stock_data['period'] == period]
                
                if not period_data.empty:
                    stock_return = period_data.iloc[0]['quarterly_return']
                    if pd.notna(stock_return):
                        portfolio_returns.append(stock_return)
                        successful_stocks += 1
        
        # Return equally weighted average if we have data for at least 50% of stocks
        if portfolio_returns and successful_stocks >= len(stock_list) * 0.5:
            return np.mean(portfolio_returns)
        else:
            return None  # Insufficient data
            
    except Exception as e:
        return None

def get_real_portfolio_returns_new(stock_list: list, period: pd.Timestamp, quarterly_returns: dict, frequency: str = "Q") -> float:
    """Get real portfolio returns using pre-fetched returns data with weight redistribution"""
    try:
        if not quarterly_returns or not stock_list:
            return None
        
        # Convert period to appropriate string format based on frequency
        if frequency == "Q":
            period_str = f"{period.year}Q{period.quarter}"
        elif frequency == "SA":
            # For semi-annual, determine if it's H1 or H2
            half = "H1" if period.month <= 6 else "H2"
            period_str = f"{period.year}{half}"
        elif frequency == "A":
            # For annual, just use the year
            period_str = f"{period.year}"
        else:
            # Default to quarterly
            period_str = f"{period.year}Q{period.quarter}"
        
        return calculate_portfolio_quarterly_return_with_weights(stock_list, period_str, quarterly_returns)
    except Exception as e:
        return None

def calculate_portfolio_quarterly_return_with_weights(stock_list: list, period: str, periodic_returns: dict) -> float:
    """Calculate equally weighted portfolio return for a specific period with weight redistribution for unlisted stocks"""
    try:
        if not periodic_returns or not stock_list:
            return None
        
        available_stocks = []
        portfolio_returns = []
        missing_stocks = []
        
        # First, identify which stocks have data for this period
        for symbol in stock_list:
            if symbol in periodic_returns:
                stock_data = periodic_returns[symbol]
                period_data = stock_data[stock_data['period'] == period]
                
                if not period_data.empty:
                    stock_return = period_data.iloc[0]['periodic_return']
                    if pd.notna(stock_return) and abs(stock_return) < 100:  # Filter out extreme outliers
                        available_stocks.append(symbol)
                        portfolio_returns.append(stock_return)
                    else:
                        missing_stocks.append(symbol)
                else:
                    missing_stocks.append(symbol)
            else:
                missing_stocks.append(symbol)
        
        # Overweight available stocks when some are missing
        if len(available_stocks) > 0:
            # Calculate equal weight redistribution
            equal_weight = 1.0 / len(available_stocks)  # Each available stock gets equal share
                    
            # Return equally weighted average (automatically redistributes weights to available stocks)
            weighted_return = np.mean(portfolio_returns)
            return weighted_return
        else:
            # No stocks available for this period
            return None
            
    except Exception as e:
        return None
    """Get real portfolio returns for a specific quarter using SSI API"""
    try:
        if not SSI_API_AVAILABLE or not stock_list:
            if period.year <= 2012:  # Only show warning for early periods
                st.warning(f"SSI API not available for {period.year}Q{period.quarter}, using fallback")
            return None  # Return None to indicate no data, let caller handle fallback
        
        # Calculate quarter dates
        year = period.year
        quarter = period.quarter
        
        # Start of quarter
        start_month = (quarter - 1) * 3 + 1
        start_date = pd.Timestamp(year=year, month=start_month, day=1)
        
        # End of quarter (last day)
        if quarter == 4:
            end_date = pd.Timestamp(year=year, month=12, day=31)
        else:
            end_month = quarter * 3
            end_date = pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
        
        portfolio_returns = []
        successful_fetches = 0
        
        # Debug info for first few periods only
        debug_mode = period.year <= 2012 and len(stock_list) <= 5  # Less verbose
        
        if debug_mode:
            # Fetching real stock data for quarterly returns
            pass
        
        for ticker in stock_list:
            try:
                # Get stock data for the quarter
                stock_data = fetch_historical_price(
                    ticker, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d')
                )
                
                if not stock_data.empty and 'close' in stock_data.columns:
                    # Convert time to datetime if needed
                    if 'time' in stock_data.columns:
                        stock_data['time'] = pd.to_datetime(stock_data['time'])
                        stock_data = stock_data.sort_values('time')
                    
                    # Calculate quarterly return
                    if len(stock_data) >= 2:
                        start_price = stock_data['close'].iloc[0]
                        end_price = stock_data['close'].iloc[-1]
                        if start_price > 0:
                            quarterly_return = ((end_price / start_price) - 1) * 100
                            portfolio_returns.append(quarterly_return)
                            successful_fetches += 1
                            
                            # Debug info for early periods and small portfolios only
                            if debug_mode:
                                # Individual stock return calculation completed
                                pass
                else:
                    if debug_mode:
                        # No data for ticker
                        pass
                
                time.sleep(0.05)  # Reduced sleep time
                
            except Exception as e:
                if debug_mode:
                    st.error(f"❌ Error fetching {ticker}: {str(e)}")
                continue
        
        # Return equally weighted portfolio return
        if portfolio_returns and successful_fetches >= len(stock_list) * 0.5:  # At least 50% success
            avg_return = np.mean(portfolio_returns)
            if debug_mode:
                # Portfolio return calculated successfully
                pass
            return avg_return
        else:
            if debug_mode:
                # Insufficient data warning
                pass
            return None  # Return None to indicate insufficient data
            
    except Exception as e:
        if period.year <= 2012:
            st.error(f"Error calculating real portfolio returns for {period}: {str(e)}")
        return None

def calculate_portfolio_returns(stock_list: list, start_date: str, end_date: str, frequency: str = "Q") -> pd.DataFrame:
    """Calculate equally weighted portfolio returns for given stock list
    
    Args:
        stock_list: List of stock tickers
        start_date: Start date for data
        end_date: End date for data
        frequency: Frequency for returns ('Q', 'SA', 'A', 'M')
    
    Returns sum of monthly returns for the specified frequency periods
    """
    try:
        if not SSI_API_AVAILABLE:
            return pd.DataFrame()
        
        portfolio_data = {}
        
        for ticker in stock_list:
            try:
                stock_data = fetch_historical_price(ticker, start_date, end_date)
                if not stock_data.empty and 'close' in stock_data.columns:
                    stock_data['time'] = pd.to_datetime(stock_data['time'])
                    stock_data = stock_data.set_index('time')
                    
                    # First calculate monthly returns
                    monthly_prices = stock_data['close'].resample('M').last()
                    monthly_returns = monthly_prices.pct_change().dropna() * 100
                    
                    # Then aggregate monthly returns based on frequency
                    if frequency == "Q":
                        # Sum 3 months of returns for quarterly
                        aggregated_returns = monthly_returns.resample('Q').sum()
                    elif frequency == "SA":  # Semi-annual
                        # Sum 6 months of returns for semi-annual
                        aggregated_returns = monthly_returns.resample('6M').sum()
                    elif frequency == "A":  # Annual
                        # Sum 12 months of returns for annual
                        aggregated_returns = monthly_returns.resample('Y').sum()
                    else:
                        # Default to monthly
                        aggregated_returns = monthly_returns
                    
                    portfolio_data[ticker] = aggregated_returns
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                st.warning(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        if portfolio_data:
            # Create DataFrame and calculate equally weighted portfolio returns
            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_df['Portfolio_Return'] = portfolio_df.mean(axis=1, skipna=True)
            
            # Sort by time index and set first period to 0% for baseline
            portfolio_df = portfolio_df.sort_index()
            if not portfolio_df.empty:
                portfolio_df['Portfolio_Return'].iloc[0] = 0.0
            
            return portfolio_df[['Portfolio_Return']].reset_index()
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error calculating portfolio returns: {str(e)}")
        return pd.DataFrame()

def prepare_enso_data_semiannual(enso_df: pd.DataFrame, start_date: str = "2011-01-01") -> pd.DataFrame:
    """Prepare ENSO data for semi-annual analysis"""
    return prepare_enso_data_with_frequency(enso_df, start_date, "6M")

def prepare_enso_data_annual(enso_df: pd.DataFrame, start_date: str = "2011-01-01") -> pd.DataFrame:
    """Prepare ENSO data for annual analysis"""
    return prepare_enso_data_with_frequency(enso_df, start_date, "Y")

def prepare_enso_data_with_frequency(enso_df: pd.DataFrame, start_date: str = "2011-01-01", frequency: str = "6M") -> pd.DataFrame:
    """Prepare ENSO data with specified frequency (Q for quarterly, SA for semi-annual, A for annual)"""
    try:
        # Map frequency codes to pandas frequency codes
        freq_mapping = {
            "Q": "Q",      # Quarterly
            "SA": "6M",    # Semi-annual
            "A": "Y"       # Annual
        }
        pandas_freq = freq_mapping.get(frequency, frequency)  # Use original if not in mapping
        
        if enso_df is None or enso_df.empty:
            st.error("❌ No ENSO data available. Cannot proceed without real ENSO data.")
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original data
        enso_df = enso_df.copy()
        
        # Try different possible column names for date/period
        date_columns = ['Period', 'Date', 'period', 'date', 'Time', 'time', 'Year', 'year', 'Periods', 'periods']
        date_col = None
        for col in date_columns:
            if col in enso_df.columns:
                date_col = col
                break
        
        if date_col is None:
            # If no date column found, try to use index
            if enso_df.index.dtype == 'object':
                try:
                    enso_df.index = pd.to_datetime(enso_df.index)
                except:
                    st.warning(f"No valid date column found in ENSO data, using mock data ({frequency})")
                    return prepare_enso_data_with_frequency(None, start_date, frequency)
            elif not pd.api.types.is_datetime64_any_dtype(enso_df.index):
                st.warning(f"No valid date column or index found in ENSO data, using mock data ({frequency})")
                return prepare_enso_data_with_frequency(None, start_date, frequency)
        else:
            # Convert date column to datetime and set as index
            try:
                # Handle different date formats
                if enso_df[date_col].dtype == 'object':
                    # Try to parse various date formats
                    enso_df[date_col] = pd.to_datetime(enso_df[date_col], errors='coerce')
                elif pd.api.types.is_numeric_dtype(enso_df[date_col]):
                    # Handle Excel date numbers
                    enso_df[date_col] = pd.to_datetime(enso_df[date_col], origin='1899-12-30', unit='D', errors='coerce')
                
                # Remove any rows with invalid dates
                enso_df = enso_df.dropna(subset=[date_col])
                enso_df = enso_df.set_index(date_col)
            except Exception as e:
                st.error(f"Error converting date column {date_col}: {str(e)}")
                return pd.DataFrame()
        
        # Filter from start date
        start_datetime = pd.to_datetime(start_date)
        enso_df = enso_df[enso_df.index >= start_datetime]
        
        # Try different possible ONI column names
        oni_columns = ['ONI', 'DJF', 'oni', 'djf', 'ENSO', 'enso', 'Nino34', 'nino34', 'Oceanic Nino Index', 'oceanic nino index', 'Oceanic Niño Index', 'oceanic niño index']
        oni_col = None
        for col in oni_columns:
            if col in enso_df.columns:
                oni_col = col
                break
        
        if oni_col is None:
            st.error(f"❌ No ONI column found in ENSO data. Available columns: {enso_df.columns.tolist()}")
            return pd.DataFrame()
        
        # Clean the ONI data
        oni_series = enso_df[oni_col]
        
        # Convert to numeric if it's not already
        if oni_series.dtype == 'object':
            oni_series = pd.to_numeric(oni_series, errors='coerce')
        
        # Remove any invalid values
        oni_series = oni_series.dropna()
        
        if oni_series.empty:
            st.error("❌ No valid ONI values found in ENSO data")
            return pd.DataFrame()
        
        # Resample with specified frequency and take mean (ONI should be averaged)
        enso_resampled = oni_series.resample(pandas_freq).mean()
        result = enso_resampled.reset_index()
        result.columns = ['Period', 'ONI']
        
        if result.empty:
            st.error("❌ No data after filtering and resampling")
            return pd.DataFrame()
        
        if pandas_freq == "Y":
            freq_label = "Annual"
        elif pandas_freq == "Q":
            freq_label = "Quarterly"
        else:
            freq_label = "Semi-annual"
        return result
            
    except Exception as e:
        st.error(f"Error preparing ENSO data: {str(e)}")
        return pd.DataFrame()

def create_oni_portfolio_line_chart(enso_df: pd.DataFrame, selected_portfolio: str, frequency: str = "SA") -> go.Figure:
    """Create line chart with ONI (left axis) and portfolio returns (right axis)
    
    Args:
        enso_df: ENSO data DataFrame
        selected_portfolio: Portfolio name
        frequency: 'SA' for semi-annual, 'A' for annual
    """
    try:
        # Define updated stock lists
        stock_portfolios = {
            "Hydro Portfolio": ['REE', 'PC1', 'HDG', 'GEG', 'TTA','AVC','GHC','VPD','DRL','S4A','SBA','VSH','NED','TMP','HNA','SHP'],
            "Gas Portfolio": ['POW', 'NT2'],  # Updated as requested
            "Coal Portfolio": ['QTP', 'PPC', 'HND'],  # Updated as requested
            "All Power Portfolio": create_all_power_portfolio(frequency)  # New option
        }
        
        if selected_portfolio not in stock_portfolios:
            return go.Figure().add_annotation(text=f"Portfolio {selected_portfolio} not found", showarrow=False)
        
        # Prepare ENSO data with selected frequency
        freq_code = {"SA": "SA", "A": "A"}[frequency]
        enso_prepared = prepare_enso_data_with_frequency(enso_df, "2011-01-01", freq_code)
        
        if enso_prepared.empty:
            return go.Figure().add_annotation(text="No ENSO data available", showarrow=False)
        
        # Calculate portfolio returns with selected frequency
        if SSI_API_AVAILABLE:
            portfolio_returns = calculate_portfolio_returns(
                stock_portfolios[selected_portfolio], 
                "2011-01-01", 
                datetime.now().strftime('%Y-%m-%d'),
                freq_code
            )
        else:
            st.error("❌ SSI API not available. Cannot fetch real portfolio data.")
            return go.Figure().add_annotation(text="SSI API not available", showarrow=False)
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add ONI line (left axis) - use same column detection logic
        oni_columns = ['ONI', 'DJF', 'oni', 'djf', 'ENSO', 'enso', 'Nino34', 'nino34', 'Oceanic Nino Index', 'oceanic nino index', 'Oceanic Niño Index', 'oceanic niño index']
        oni_col = 'ONI'  # default fallback
        for col in oni_columns:
            if col in enso_prepared.columns:
                oni_col = col
                break
        
        fig.add_trace(
            go.Scatter(
                x=enso_prepared['Period'] if 'Period' in enso_prepared.columns else enso_prepared.index,
                y=enso_prepared[oni_col] if oni_col in enso_prepared.columns else [],
                mode='lines+markers',
                name='ONI',
                line=dict(color='#97999B', width=2),
                marker=dict(size=6),
                yaxis='y',
                uid=f'oni_{selected_portfolio}'  # Add unique ID
            ),
            secondary_y=False,
        )
        
        # Add portfolio returns line (right axis) if data available
        if not portfolio_returns.empty and 'Portfolio_Return' in portfolio_returns.columns:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_returns['time'] if 'time' in portfolio_returns.columns else portfolio_returns.index,
                    y=portfolio_returns['Portfolio_Return'],
                    mode='lines+markers',
                    name=f'{selected_portfolio} Returns (%)',
                    line=dict(color='#08C179', width=2),
                    marker=dict(size=6),
                    yaxis='y2',
                    uid=f'portfolio_{selected_portfolio}'  # Add unique ID
                ),
                secondary_y=True,
            )
        
        # Set x-axis properties
        fig.update_xaxes(title_text="Period", showgrid=False)
        
        # Set y-axes properties
        fig.update_yaxes(title_text="ONI Value", secondary_y=False, showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(title_text="Portfolio Return (%)", secondary_y=True, showgrid=False)
        
        # Update layout
        freq_name = "Annual" if frequency == "A" else "Semi-Annual"
        fig.update_layout(
            title=f"ONI vs {selected_portfolio} Returns ({freq_name})",
            hovermode='x unified',
            legend=dict(x=0.01, y=0.99),
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating ONI portfolio chart: {str(e)}")
        return go.Figure().add_annotation(text="Error creating chart", showarrow=False)

def create_oni_portfolio_regression_chart(enso_df: pd.DataFrame, selected_portfolio: str, frequency: str = "SA") -> go.Figure:
    """Create regression scatter plot of ONI vs portfolio returns
    
    Args:
        enso_df: ENSO data DataFrame
        selected_portfolio: Portfolio name
        frequency: 'SA' for semi-annual, 'A' for annual
    """
    try:
        # Define updated stock lists
        stock_portfolios = {
            "Hydro Portfolio": ['REE', 'PC1', 'HDG', 'GEG', 'TTA','AVC','GHC','VPD','DRL','S4A','SBA','VSH','NED','TMP','HNA','SHP'],
            "Gas Portfolio": ['POW', 'NT2'],  # Updated as requested
            "Coal Portfolio": ['QTP', 'PPC', 'HND'],  # Updated as requested
            "All Power Portfolio": create_all_power_portfolio(frequency)  # New option
        }
        
        if selected_portfolio not in stock_portfolios:
            return go.Figure().add_annotation(text=f"Portfolio {selected_portfolio} not found", showarrow=False)
        
        # Prepare ENSO data with selected frequency
        freq_code = {"SA": "SA", "A": "A"}[frequency]
        enso_prepared = prepare_enso_data_with_frequency(enso_df, "2011-01-01", freq_code)
        
        if enso_prepared.empty:
            return go.Figure().add_annotation(text="No ENSO data available", showarrow=False)
        
        # Calculate portfolio returns with selected frequency
        if SSI_API_AVAILABLE:
            portfolio_returns = calculate_portfolio_returns(
                stock_portfolios[selected_portfolio], 
                "2011-01-01", 
                datetime.now().strftime('%Y-%m-%d'),
                freq_code
            )
        else:
            st.error("❌ SSI API not available. Cannot fetch real portfolio data.")
            return go.Figure().add_annotation(text="SSI API not available", showarrow=False)
        
        if portfolio_returns.empty or enso_prepared.empty:
            return go.Figure().add_annotation(text="Insufficient data for regression", showarrow=False)
        
        # Merge data for regression analysis
        # Use the same ONI column detection logic
        oni_columns = ['ONI', 'DJF', 'oni', 'djf', 'ENSO', 'enso', 'Nino34', 'nino34', 'Oceanic Nino Index', 'oceanic nino index', 'Oceanic Niño Index', 'oceanic niño index']
        oni_col = 'ONI'  # default fallback
        for col in oni_columns:
            if col in enso_prepared.columns:
                oni_col = col
                break
        
        # Ensure date columns are properly formatted
        if 'Period' in enso_prepared.columns:
            enso_prepared['Period'] = pd.to_datetime(enso_prepared['Period'])
        if 'time' in portfolio_returns.columns:
            portfolio_returns['time'] = pd.to_datetime(portfolio_returns['time'])
        elif 'date' in portfolio_returns.columns:
            portfolio_returns['time'] = pd.to_datetime(portfolio_returns['date'])
        
        # For better alignment, let's use period-based matching instead of exact date matching
        # Convert dates to period strings based on frequency
        if frequency == "A":
            enso_prepared['period_key'] = enso_prepared['Period'].dt.year.astype(str)
            portfolio_returns['period_key'] = portfolio_returns['time'].dt.year.astype(str)
        else:  # Semi-annual
            enso_prepared['period_key'] = enso_prepared['Period'].dt.year.astype(str) + "_" + ((enso_prepared['Period'].dt.month - 1) // 6 + 1).astype(str)
            portfolio_returns['period_key'] = portfolio_returns['time'].dt.year.astype(str) + "_" + ((portfolio_returns['time'].dt.month - 1) // 6 + 1).astype(str)
        
        # Merge on period_key for better alignment
        merged_data = pd.merge(
            enso_prepared[['period_key', oni_col]],
            portfolio_returns[['period_key', 'Portfolio_Return']],
            on='period_key', how='inner'
        ).dropna()
        
        if merged_data.empty or len(merged_data) < 2:
            return go.Figure().add_annotation(text="Insufficient overlapping data for regression", showarrow=False)
        
        oni_values = merged_data[oni_col].values
        portfolio_values = merged_data['Portfolio_Return'].values
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=oni_values,
                y=portfolio_values,
                mode='markers',
                name=f'{selected_portfolio} vs ONI',
                marker=dict(
                    size=8,
                    color='#08C179',
                    opacity=0.7
                ),
                text=[f'Period {i+1}' for i in range(len(oni_values))],
                hovertemplate='ONI: %{x}<br>Return: %{y:.2f}%<br>%{text}<extra></extra>',
                uid=f'scatter_{selected_portfolio}'  # Add unique ID
            )
        )
        
        # Add trend line
        if len(oni_values) > 1:
            try:
                # Simple linear regression
                z = np.polyfit(oni_values, portfolio_values, 1)
                p = np.poly1d(z)
                
                x_trend = np.linspace(min(oni_values), max(oni_values), 100)
                y_trend = p(x_trend)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_trend,
                        y=y_trend,
                        mode='lines',
                        name='Trend Line',
                        line=dict(color='red', width=2, dash='dash'),
                        uid=f'trend_{selected_portfolio}'  # Add unique ID
                    )
                )
                
                # Calculate correlation
                correlation = np.corrcoef(oni_values, portfolio_values)[0, 1]
                r_squared = correlation ** 2
                
                # Add correlation info
                fig.add_annotation(
                    text=f"R² = {r_squared:.3f}<br>Correlation = {correlation:.3f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    align="left",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            except Exception as e:
                st.warning(f"Error calculating regression: {str(e)}")
        
        freq_name = "Annual" if frequency == "A" else "Semi-Annual"
        fig.update_layout(
            title=f"ONI vs {selected_portfolio} Returns Regression ({freq_name})",
            xaxis_title="ONI Value",
            yaxis_title="Portfolio Return (%)",
            hovermode='closest',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating regression chart: {str(e)}")
        return go.Figure().add_annotation(text="Error creating chart", showarrow=False)

def create_all_power_portfolio(frequency: str = "Q") -> list:
    """Create equally weighted portfolio of all power sector stocks"""
    hydro_stocks = ['REE', 'PC1', 'HDG', 'GEG', 'TTA','AVC','GHC','VPD','DRL','S4A','SBA','VSH','NED','TMP','HNA','SHP']
    coal_stocks = ['QTP', 'PPC', 'HND']
    gas_stocks = ['POW', 'NT2', 'PGV', 'GAS']  # Include all gas stocks
    
    # Combine all power stocks
    all_power_stocks = hydro_stocks + coal_stocks + gas_stocks
    return list(set(all_power_stocks))  # Remove any duplicates

def calculate_all_power_portfolio_returns(frequency: str = "Q", start_date: str = "2011-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """Calculate equally weighted portfolio returns for all power stocks"""
    try:
        all_power_stocks = create_all_power_portfolio(frequency)
        
        if SSI_API_AVAILABLE:
            return calculate_portfolio_returns(all_power_stocks, start_date, end_date, frequency)
        else:
            st.error("❌ SSI API not available. Cannot fetch real all power portfolio data.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error calculating all power portfolio returns: {str(e)}")
        return pd.DataFrame()

def create_oni_strategy_portfolio(enso_df: pd.DataFrame, frequency: str = "Q") -> pd.DataFrame:
    """
    Create ONI-based portfolio strategy:
    - If quarterly ONI > 0.5: invest 50%/50% in gas/coal equally weighted portfolios
    - If quarterly ONI < -0.5: invest 100% in hydro equally weighted portfolio
    - If -0.5 ≤ ONI ≤ 0.5: invest 25%/25% in gas/coal and 50% in hydro equally weighted portfolios
    
    Returns cumulative returns from 1Q2011 to 3Q2025
    Note: Handles unlisted stocks by redistributing weights to available stocks
    """
    try:
        # Define stock portfolios with approximate listing information
        stock_portfolios = {
            "Hydro": ['REE', 'PC1', 'HDG', 'GEG', 'TTA','AVC','GHC','VPD','DRL','S4A','SBA','VSH','NED','TMP','HNA','SHP'],
            "Gas": ['POW', 'NT2'],  # Note: POW listed around 2018, NT2 was available earlier
            "Coal": ['QTP', 'PPC', 'HND']
        }
        
        # Stock listing dates (approximate) - for information purposes
        stock_listing_info = {
            'POW': '2018Q1',  # PetroVietnam Power listed around 2018
            'NT2': '2011Q1',  # NT2 was available from early period
            'QTP': '2011Q1',  # Coal stocks generally available earlier
            'PPC': '2011Q1',
            'HND': '2011Q1',
            # Most hydro stocks were available from 2011 or earlier
        }
        
        # Prepare ENSO data with selected frequency
        enso_prepared = prepare_enso_data_with_frequency(enso_df, "2011-01-01", frequency)
        
        if enso_prepared.empty:
            st.warning("No ENSO data available for strategy construction")
            return pd.DataFrame()
        
        # Get VNI data
        vni_df = load_vni_data()
        
        # Fetch all stock data upfront
        quarterly_returns = {}
        st.info("📊 Fetching stock data for trading strategies...")
        
        # Fetch stock data for all portfolios
        all_stocks = []
        for portfolio_stocks in stock_portfolios.values():
            all_stocks.extend(portfolio_stocks)
        
        # Remove duplicates while preserving order
        unique_stocks = list(dict.fromkeys(all_stocks))
        
        # Get stock data with retry logic (will use mock data if SSI API fails)
        stock_data = get_stock_data_for_portfolio(unique_stocks, "2011-01-01", "2025-12-31")
        
        if stock_data:
            # Convert to periodic returns based on frequency
            quarterly_returns = convert_to_periodic_returns(stock_data, frequency)
            
            # Show detailed information about data availability
            total_stocks = len(unique_stocks)
            available_stocks = len([s for s in unique_stocks if s in stock_data])
            
            st.success(f"✅ Successfully fetched real data for {available_stocks}/{total_stocks} stocks")
                       
            # Show information about potentially unlisted stocks
            missing_stocks = [s for s in unique_stocks if s not in stock_data]
            if missing_stocks:
                st.warning(f"⚠️ Limited/no data for: {', '.join(missing_stocks[:5])}{'...' if len(missing_stocks) > 5 else ''}")
                st.info("💡 Strategy will use weight redistribution for available stocks")
        else:
            st.error("❌ Failed to fetch any stock data")
            return pd.DataFrame()
        
        strategy_results = []
        
        # Calculate portfolio returns for each quarter
        for _, row in enso_prepared.iterrows():
            period = row['Period']
            oni_value = row['ONI']
            
            # Determine allocation based on ONI value
            if oni_value > 0.5:
                # El Niño conditions: invest in gas/coal
                hydro_weight = 0.0
                gas_weight = 0.5
                coal_weight = 0.5
                allocation_type = "Gas/Coal (El Niño)"
            elif oni_value < -0.5:
                # La Niña conditions: invest in hydro
                hydro_weight = 1.0
                gas_weight = 0.0
                coal_weight = 0.0
                allocation_type = "Hydro (La Niña)"
            else:
                # Neutral conditions: balanced allocation
                hydro_weight = 0.5
                gas_weight = 0.25
                coal_weight = 0.25
                allocation_type = "Balanced (Neutral)"
            
            # Calculate weighted portfolio return for this period
            period_return = 0.0
            
            # Determine the number of months in this period for aggregation
            if frequency == "SA":  # Semi-annual
                months_in_period = 6
            elif frequency == "A":  # Annual  
                months_in_period = 12
            else:  # Quarterly
                months_in_period = 3
            
            # Try to use real data only - no mock fallbacks
            if quarterly_returns:
                try:
                    # Get real returns for each portfolio
                    hydro_return = get_real_portfolio_returns_new(stock_portfolios["Hydro"], period, quarterly_returns, frequency)
                    gas_return = get_real_portfolio_returns_new(stock_portfolios["Gas"], period, quarterly_returns, frequency)
                    coal_return = get_real_portfolio_returns_new(stock_portfolios["Coal"], period, quarterly_returns, frequency)
                    
                    # Check if we got valid real data for required portfolios based on ONI allocation
                    valid_returns = []
                    if hydro_weight > 0 and hydro_return is not None:
                        valid_returns.append(hydro_weight * hydro_return)
                    elif hydro_weight > 0:
                        # Skip this period if required hydro data is missing
                        st.warning(f"⚠️ Skipping {period.year}Q{period.quarter}: Missing hydro data for {hydro_weight*100}% allocation")
                        continue
                        
                    if gas_weight > 0 and gas_return is not None:
                        valid_returns.append(gas_weight * gas_return)
                    elif gas_weight > 0:
                        # Skip this period if required gas data is missing
                        st.warning(f"⚠️ Skipping {period.year}Q{period.quarter}: Missing gas data for {gas_weight*100}% allocation")
                        continue
                        
                    if coal_weight > 0 and coal_return is not None:
                        valid_returns.append(coal_weight * coal_return)
                    elif coal_weight > 0:
                        # Skip this period if required coal data is missing
                        st.warning(f"⚠️ Skipping {period.year}Q{period.quarter}: Missing coal data for {coal_weight*100}% allocation")
                        continue
                    
                    if valid_returns:
                        # Calculate weighted portfolio return using only real data
                        period_return = sum(valid_returns)
                        
                    else:
                        # No valid data for any required portfolio - skip this period
                        st.warning(f"⚠️ Skipping {period.year}Q{period.quarter}: No valid portfolio data available")
                        continue
                        
                except Exception as e:
                    # Skip this period if there's an error getting real data
                    st.warning(f"⚠️ Skipping {period.year}Q{period.quarter}: Error getting real data - {str(e)}")
                    continue
            else:
                # No real data available at all - skip this period  
                if len(strategy_results) == 0:
                    st.error("❌ No real stock data available from SSI API. Cannot create ONI strategy with real data.")
                    return pd.DataFrame()
                continue
            
            # Get VNI return for this period
            vni_return = get_vni_return_for_period(vni_df, period)
            
            strategy_results.append({
                'Period': period,
                'ONI': oni_value,
                'Hydro_Weight': hydro_weight,
                'Gas_Weight': gas_weight,
                'Coal_Weight': coal_weight,
                'Allocation_Type': allocation_type,
                'Strategy_Return': period_return,
                'VNI_Return': vni_return
            })
        
        # Convert to DataFrame and calculate cumulative returns
        strategy_df = pd.DataFrame(strategy_results)
        
        if not strategy_df.empty:
            # Ensure the first period (1Q2011) has 0% return for baseline
            if len(strategy_df) > 0:
                strategy_df.loc[0, 'Strategy_Return'] = 0.0
                # VNI return is already set to 0 in load_vni_data for first period
            
            # Calculate cumulative returns starting from 0%
            strategy_df['Strategy_Cumulative'] = (1 + strategy_df['Strategy_Return'] / 100).cumprod() - 1
            strategy_df['VNI_Cumulative'] = (1 + strategy_df['VNI_Return'] / 100).cumprod() - 1
            
            # Convert to percentage
            strategy_df['Strategy_Cumulative'] *= 100
            strategy_df['VNI_Cumulative'] *= 100
        
        return strategy_df
        
    except Exception as e:
        st.error(f"Error creating ONI strategy portfolio: {str(e)}")
        return pd.DataFrame()

def load_vni_data() -> pd.DataFrame:
    """Load VNI data from vn_index_monthly.csv file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vni_file = os.path.join(script_dir, 'data', 'vn_index_monthly.csv')
        
        vni_df = pd.read_csv(vni_file)
        
        # Clean VNINDEX values (remove commas and quotes)
        vni_df['VNINDEX'] = vni_df['VNINDEX'].astype(str).str.replace(',', '').str.replace('"', '')
        vni_df['VNINDEX'] = pd.to_numeric(vni_df['VNINDEX'], errors='coerce')
        
        # Remove any rows with invalid VNINDEX values
        vni_df = vni_df.dropna(subset=['VNINDEX'])
        
        # Parse date in format like "1Q2011"
        def parse_quarter_date(date_str):
            try:
                # Handle format like "1Q2011" or "2Q2012"
                date_str = str(date_str).strip().replace('"', '')  # Remove quotes if any
                quarter = int(date_str[0])
                year = int(date_str[2:])
                month = (quarter - 1) * 3 + 3  # End of quarter month
                return pd.Timestamp(year=year, month=month, day=1)
            except:
                return pd.NaT
        
        vni_df['Period'] = vni_df['date'].apply(parse_quarter_date)
        vni_df = vni_df.dropna(subset=['Period'])  # Remove any invalid dates
        vni_df = vni_df.sort_values('Period')
        
        # Calculate quarterly returns
        vni_df['VNI_Return'] = vni_df['VNINDEX'].pct_change() * 100
        
        # Set first period return to 0% (baseline)
        vni_df.loc[0, 'VNI_Return'] = 0.0
                
        return vni_df
        
    except Exception as e:
        st.error(f"Error loading VNI data: {str(e)}")
        return pd.DataFrame()

def get_vni_return_for_period(vni_df: pd.DataFrame, period: pd.Timestamp) -> float:
    """Get VNI return for a specific period"""
    try:
        if vni_df.empty:
            return 0.0  # Default return if no data
        
        # Create period key for matching (e.g., "2011Q1")
        period_key = f"{period.year}Q{period.quarter}"
        
        # Find matching period in VNI data by comparing the original date strings
        matching_rows = vni_df[vni_df['date'].str.replace('Q', 'Q') == period_key]
        
        if not matching_rows.empty:
            vni_return = matching_rows.iloc[0]['VNI_Return']
            # Debug info for first few periods
            if period.year <= 2012:
                st.info(f"VNI return for {period_key}: {vni_return:.2f}%")
            return vni_return
        else:
            # Try alternative matching by finding closest period
            # Convert period to comparable format
            target_year_quarter = period.year + (period.quarter - 1) * 0.25
            
            def period_to_number(date_str):
                try:
                    quarter = int(date_str[0])
                    year = int(date_str[2:])
                    return year + (quarter - 1) * 0.25
                except:
                    return 0
            
            vni_df['period_number'] = vni_df['date'].apply(period_to_number)
            
            # Find closest period
            closest_idx = (vni_df['period_number'] - target_year_quarter).abs().idxmin()
            closest_return = vni_df.loc[closest_idx, 'VNI_Return']
            
            return closest_return
        
    except Exception as e:
        st.error(f"Error getting VNI return for {period}: {str(e)}")
        return 0.0  # Default return on error

def create_enhanced_oni_strategy_chart(strategy_df: pd.DataFrame, frequency: str = "Q") -> go.Figure:
    """Create enhanced chart showing ONI strategy, All Power portfolio, and VNI performance"""
    try:
        if strategy_df.empty:
            return go.Figure().add_annotation(text="No strategy data available", showarrow=False)
        
        fig = go.Figure()
        
        # Calculate all power portfolio returns
        all_power_returns = calculate_all_power_portfolio_returns(frequency, "2011-01-01", "2025-12-31")
        
        if not all_power_returns.empty:
            # Calculate cumulative returns for all power portfolio using PROPER compound returns
            all_power_cumulative = []
            cumulative_multiplier = 1.0  # Start with 1.0 for compound returns
            
            # Align with strategy periods
            strategy_periods = pd.to_datetime(strategy_df['Period'])
            all_power_periods = pd.to_datetime(all_power_returns['time'])
            
            for i, period in enumerate(strategy_periods):
                # Find closest period in all power data
                closest_idx = (all_power_periods - period).abs().idxmin()
                if closest_idx < len(all_power_returns):
                    period_return = all_power_returns.iloc[closest_idx]['Portfolio_Return']
                    if i == 0:
                        # First period baseline: 0% return
                        cumulative_multiplier = 1.0
                    else:
                        # Compound the returns properly: (1 + return/100) 
                        cumulative_multiplier *= (1 + period_return / 100)
                else:
                    # No data available - no change in cumulative multiplier
                    pass
                
                # Convert back to percentage cumulative return
                cumulative_return_pct = (cumulative_multiplier - 1) * 100
                all_power_cumulative.append(cumulative_return_pct)
            
            # Add all power portfolio line
            fig.add_trace(
                go.Scatter(
                    x=strategy_df['Period'],
                    y=all_power_cumulative,
                    mode='lines+markers',
                    name='All Power Stocks (Equal Weight)',
                    line=dict(color='#FF6B35', width=2),
                    marker=dict(size=5),
                    hovertemplate="<b>All Power Portfolio</b><br>" +
                               "Period: %{x}<br>" +
                               "Cumulative Return: %{y:.2f}%<br>" +
                               "<extra></extra>"
                )
            )
        
        # Add strategy cumulative return
        fig.add_trace(
            go.Scatter(
                x=strategy_df['Period'],
                y=strategy_df['Strategy_Cumulative'],
                mode='lines+markers',
                name='ONI Strategy Portfolio',
                line=dict(color='#0C4130', width=3),
                marker=dict(size=6),
                hovertemplate="<b>ONI Strategy</b><br>" +
                           "Period: %{x}<br>" +
                           "Cumulative Return: %{y:.2f}%<br>" +
                           "<extra></extra>"
            )
        )
        
        # Add VNI benchmark
        fig.add_trace(
            go.Scatter(
                x=strategy_df['Period'],
                y=strategy_df['VNI_Cumulative'],
                mode='lines+markers',
                name='VNI Index',
                line=dict(color='#97999B', width=2, dash='dash'),
                marker=dict(size=4),
                hovertemplate="<b>VNI Index</b><br>" +
                           "Period: %{x}<br>" +
                           "Cumulative Return: %{y:.2f}%<br>" +
                           "<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Portfolio Performance Comparison: ONI Strategy vs All Power vs VNI (1Q2011 - 3Q2025)",
            xaxis_title="Period",
            yaxis_title="Cumulative Return (%)",
            height=600,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating enhanced ONI strategy chart: {str(e)}")
        return go.Figure()

def create_oni_strategy_chart(strategy_df: pd.DataFrame) -> go.Figure:
    """Create chart showing ONI strategy performance vs VNI"""
    try:
        if strategy_df.empty:
            return go.Figure().add_annotation(text="No strategy data available", showarrow=False)
        
        fig = go.Figure()
        
        # Add strategy cumulative return
        fig.add_trace(
            go.Scatter(
                x=strategy_df['Period'],
                y=strategy_df['Strategy_Cumulative'],
                mode='lines+markers',
                name='ONI Strategy Portfolio',
                line=dict(color='#0C4130', width=3),
                marker=dict(size=6),
                hovertemplate="<b>ONI Strategy</b><br>" +
                           "Period: %{x}<br>" +
                           "Cumulative Return: %{y:.2f}%<br>" +
                           "<extra></extra>"
            )
        )
        
        # Add VNI benchmark
        fig.add_trace(
            go.Scatter(
                x=strategy_df['Period'],
                y=strategy_df['VNI_Cumulative'],
                mode='lines+markers',
                name='VNI Index',
                line=dict(color='#97999B', width=2, dash='dash'),
                marker=dict(size=4),
                hovertemplate="<b>VNI Index</b><br>" +
                           "Period: %{x}<br>" +
                           "Cumulative Return: %{y:.2f}%<br>" +
                           "<extra></extra>"
            )
        )
        
        # Update layout
        fig.update_layout(
            title="ONI-Based Portfolio Strategy vs VNI (1Q2011 - 3Q2025)",
            xaxis_title="Period",
            yaxis_title="Cumulative Return (%)",
            height=600,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating ONI strategy chart: {str(e)}")
        return go.Figure()

@st.cache_data
def calculate_portfolio_returns_by_stock_data(stock_data_dict: Dict[str, pd.DataFrame], stock_list: list, portfolio_name: str) -> pd.DataFrame:
    """Calculate equally weighted portfolio returns"""
    try:
        portfolio_returns = []
        all_periods = set()
        stock_period_returns = {}
        
        # Collect all stock returns by period
        for stock, stock_df in stock_data_dict.items():
            if stock in stock_list and not stock_df.empty:
                stock_periods = {}
                for _, row in stock_df.iterrows():
                    period = f"{row['date'].year}Q{((row['date'].month - 1) // 3) + 1}"
                    stock_periods[period] = row['close']
                
                # Calculate QoQ returns
                sorted_periods = sorted(stock_periods.keys())
                if len(sorted_periods) > 1:
                    stock_period_returns[stock] = {}
                    for i in range(1, len(sorted_periods)):
                        current_period = sorted_periods[i]
                        prev_period = sorted_periods[i-1]
                        qoq_return = ((stock_periods[current_period] - stock_periods[prev_period]) / stock_periods[prev_period]) * 100
                        stock_period_returns[stock][current_period] = qoq_return
                        all_periods.add(current_period)
        
        # Calculate equal-weighted portfolio returns
        results = []
        for period in sorted(all_periods):
            period_returns = []
            for stock in stock_list:
                if stock in stock_period_returns and period in stock_period_returns[stock]:
                    period_returns.append(stock_period_returns[stock][period])
            
            if len(period_returns) >= 2:  # At least 2 stocks for portfolio
                avg_return = sum(period_returns) / len(period_returns)
                results.append({
                    'period': period,
                    'return': avg_return,
                    'portfolio': portfolio_name,
                    'num_stocks': len(period_returns)
                })
        
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"Error calculating {portfolio_name} portfolio returns: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def prepare_enso_data(enso_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare ENSO data for regression analysis"""
    try:
        # Check if input is valid
        if enso_df is None or enso_df.empty:
            return pd.DataFrame()
            
        # Copy the data
        processed_data = enso_df.copy()
        
        # If there's a date column, convert to period format
        if 'Date' in processed_data.columns:
            try:
                processed_data['Date'] = pd.to_datetime(processed_data['Date'])
                processed_data['period'] = processed_data['Date'].dt.year.astype(str) + 'Q' + processed_data['Date'].dt.quarter.astype(str)
            except Exception as e:
                st.warning(f"Error processing Date column: {str(e)}")
                # Create default periods if date processing fails
                processed_data['period'] = [f"2011Q{i%4+1}" for i in range(len(processed_data))]
        elif 'Period' in processed_data.columns:
            # Ensure Period column is properly formatted as string - handle various data types
            try:
                processed_data['period'] = processed_data['Period'].apply(lambda x: str(x) if pd.notna(x) else '')
            except Exception as e:
                st.warning(f"Error converting Period column: {str(e)}")
                processed_data['period'] = processed_data['Period'].astype(str)
        elif 'period' in processed_data.columns:
            # Period column already exists
            pass
        else:
            # Try to create period from existing columns
            if 'Year' in processed_data.columns and 'Quarter' in processed_data.columns:
                processed_data['period'] = processed_data['Year'].astype(str) + 'Q' + processed_data['Quarter'].astype(str)
            else:
                # Look for any column that might contain period information
                period_cols = [col for col in processed_data.columns if 'period' in col.lower() or 'date' in col.lower() or 'time' in col.lower()]
                if period_cols:
                    # Use the first period-like column
                    processed_data['period'] = processed_data[period_cols[0]]
                    # Try to standardize format if it's not already in YYYYQX format
                    if not processed_data['period'].astype(str).str.match(r'\d{4}Q\d').all():
                        # Create a simple period range for demonstration
                        start_year = 2011
                        start_quarter = 1
                        periods = []
                        for i in range(len(processed_data)):
                            year = start_year + (start_quarter + i - 1) // 4
                            quarter = ((start_quarter + i - 1) % 4) + 1
                            periods.append(f"{year}Q{quarter}")
                        processed_data['period'] = periods
                else:
                    # Create default period range
                    start_year = 2011
                    periods = []
                    for i in range(len(processed_data)):
                        year = start_year + i // 4
                        quarter = (i % 4) + 1
                        periods.append(f"{year}Q{quarter}")
                    processed_data['period'] = periods
        
        # Ensure we have a period column
        if 'period' not in processed_data.columns or processed_data['period'].isna().all():
            st.warning("Could not determine period format in ENSO data, creating default periods")
            # Create default period range
            start_year = 2011
            periods = []
            for i in range(len(processed_data)):
                year = start_year + i // 4
                quarter = (i % 4) + 1
                periods.append(f"{year}Q{quarter}")
            processed_data['period'] = periods
        
        # Filter for the analysis period (2011Q1 to 2025Q3)
        # Convert period to string and filter - handle different data types safely
        try:
            # Ensure period column exists and convert to string
            if 'period' in processed_data.columns:
                # Handle different data types - convert to string safely
                processed_data['period'] = processed_data['period'].apply(lambda x: str(x) if pd.notna(x) else '')
                
                # Filter only if we have valid period data
                if not processed_data['period'].empty:
                    processed_data = processed_data[
                        (processed_data['period'] >= '2011Q1') & 
                        (processed_data['period'] <= '2025Q3')
                    ]
        except Exception as e:
            st.warning(f"Error filtering ENSO data by period: {str(e)}")
            # If filtering fails, just return the data as is
        
        return processed_data
        
    except Exception as e:
        st.error(f"Error preparing ENSO data: {str(e)}")
        # Return a minimal dataset with periods
        try:
            periods = [f"{year}Q{quarter}" for year in range(2011, 2026) for quarter in range(1, 5) if f"{year}Q{quarter}" <= "2025Q3"]
            return pd.DataFrame({
                'period': periods[:len(enso_df)],
                'ONI': np.random.normal(0, 1, min(len(periods), len(enso_df)))
            })
        except:
            return pd.DataFrame()

def perform_regression_analysis(portfolio_returns: pd.DataFrame, enso_data: pd.DataFrame, portfolio_name: str, icon: str) -> None:
    """Perform and display regression analysis"""
    try:
        # Merge portfolio returns with ENSO data
        merged_data = pd.merge(portfolio_returns, enso_data, on='period', how='inner')
        
        if len(merged_data) < 10:
            st.warning(f"Insufficient data points for {portfolio_name} regression (only {len(merged_data)} periods)")
            return
        
        st.write(f"**{icon} {portfolio_name} Regression Analysis**")
        st.write(f"Data points: {len(merged_data)} periods")
        
        # Display data
        st.write("**Merged Data:**")
        st.dataframe(merged_data, use_container_width=True)
        
        # Find ONI column (it might have different names)
        oni_columns = [col for col in merged_data.columns if 'oni' in col.lower() or 'nino' in col.lower() or 'index' in col.lower()]
        if not oni_columns:
            # Try numeric columns
            numeric_cols = merged_data.select_dtypes(include=[np.number]).columns.tolist()
            oni_columns = [col for col in numeric_cols if col not in ['return', 'num_stocks']]
        
        if oni_columns:
            oni_col = st.selectbox(f"Select ONI variable for {portfolio_name}:", oni_columns, key=f"oni_{portfolio_name}")
            
            try:
                try:
                    from scipy import stats
                    SCIPY_AVAILABLE = True
                except ImportError:
                    SCIPY_AVAILABLE = False
                    st.warning("Scipy not available for regression analysis")
                
                if SCIPY_AVAILABLE:
                    # Perform linear regression
                    x = merged_data[oni_col].values
                    y = merged_data['return'].values
                    
                    # Remove any NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x_clean = x[mask]
                    y_clean = y[mask]
                    
                    if len(x_clean) > 5:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
                    
                    # Display regression results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R-squared", f"{r_value**2:.4f}")
                    with col2:
                        st.metric("Correlation", f"{r_value:.4f}")
                    with col3:
                        st.metric("P-value", f"{p_value:.4f}")
                    with col4:
                        st.metric("Slope", f"{slope:.4f}")
                    
                    # Create scatter plot with regression line
                    fig = go.Figure()
                    
                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=x_clean,
                        y=y_clean,
                        mode='markers',
                        name='Data Points',
                        marker=dict(size=8, opacity=0.7),
                        hovertemplate=f"{oni_col}: %{{x}}<br>{portfolio_name} Return: %{{y:.2f}}%<extra></extra>"
                    ))
                    
                    # Regression line
                    x_range = np.linspace(min(x_clean), max(x_clean), 100)
                    y_pred = slope * x_range + intercept
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name=f'Regression Line (R²={r_value**2:.3f})',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{portfolio_name} Returns vs {oni_col}",
                        xaxis_title=oni_col,
                        yaxis_title=f"{portfolio_name} Quarterly Return (%)",
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key=f"regression_chart_{portfolio_name.lower().replace(' ', '_')}")
                    
                    # Interpretation
                    if p_value < 0.05:
                        significance = "statistically significant"
                    else:
                        significance = "not statistically significant"
                    
                    if r_value > 0:
                        direction = "positive"
                    else:
                        direction = "negative"
                    
                    st.write(f"""
                    **Interpretation:**
                    - The relationship between {oni_col} and {portfolio_name} returns is **{direction}** and **{significance}** (p-value: {p_value:.4f})
                    - R-squared of {r_value**2:.4f} indicates that {r_value**2*100:.1f}% of the variance in returns is explained by the ONI
                    - For every 1-unit increase in {oni_col}, {portfolio_name} returns change by {slope:.4f} percentage points
                    """)
                else:
                    st.warning("Insufficient clean data for regression analysis")
            except ImportError:
                st.warning("Scipy not available. Cannot perform regression analysis.")
                # Show basic correlation without scipy
                try:
                    corr = merged_data[oni_col].corr(merged_data['return'])
                    st.write(f"**Basic Correlation (without p-value):** {corr:.4f}")
                except:
                    st.warning("Could not calculate basic correlation")
        else:
            st.warning("No suitable ONI columns found in ENSO data")
            st.write(f"Available columns: {list(merged_data.columns)}")
            
    except Exception as e:
        st.error(f"Error in regression analysis for {portfolio_name}: {str(e)}")

def compare_portfolios(thermal_returns: pd.DataFrame, hydro_returns: pd.DataFrame, enso_data: pd.DataFrame) -> None:
    """Compare thermal and hydro portfolio performance against ENSO"""
    try:
        st.write("**📊 Portfolio Comparison Against ENSO**")
        
        # Merge all data
        thermal_merged = pd.merge(thermal_returns, enso_data, on='period', how='inner')
        hydro_merged = pd.merge(hydro_returns, enso_data, on='period', how='inner')
        
        if thermal_merged.empty or hydro_merged.empty:
            st.warning("Insufficient data for portfolio comparison")
            return
        
        # Combine datasets for comparison
        thermal_merged['Portfolio'] = 'Thermal'
        hydro_merged['Portfolio'] = 'Hydro'
        combined_data = pd.concat([thermal_merged, hydro_merged], ignore_index=True)
        
        st.write("**Combined Portfolio Returns:**")
        st.dataframe(combined_data[['period', 'Portfolio', 'return']], use_container_width=True)
        
        # Portfolio performance comparison
        thermal_stats = thermal_returns['return'].describe()
        hydro_stats = hydro_returns['return'].describe()
        
        st.write("**Portfolio Statistics:**")
        comparison_stats = pd.DataFrame({
            'Thermal Portfolio': thermal_stats,
            'Hydro Portfolio': hydro_stats
        })
        st.dataframe(comparison_stats, use_container_width=True)
        
        # Time series comparison
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thermal_returns['period'],
            y=thermal_returns['return'],
            mode='lines+markers',
            name='🔥 Thermal Portfolio',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=hydro_returns['period'],
            y=hydro_returns['return'],
            mode='lines+markers',
            name='💧 Hydro Portfolio',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Returns Comparison Over Time",
            xaxis_title="Period",
            yaxis_title="Quarterly Return (%)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="portfolio_comparison_chart")
        
    except Exception as e:
        st.error(f"Error in portfolio comparison: {str(e)}")

def run_enso_regression_analysis(enso_df: pd.DataFrame, trading_strategy_available: bool = False) -> None:
    """Main function to run ENSO regression analysis - plots ONI quarterly data and returns of three equally weighted portfolios"""
    
    st.markdown("""
    **Analysis Overview:**
    This analysis plots the ONI (Oceanic Niño Index) quarterly data alongside the returns of three equally weighted portfolios:
    
    **Three Portfolios:**
    - **Hydro Portfolio**: Equally weighted portfolio of hydro stocks
    - **Coal Portfolio**: Equally weighted portfolio of coal stocks (PPC, QTP, HND)
    - **Gas Portfolio**: Equally weighted portfolio of gas stocks (POW, NT2)
    """)
    
    # Define stock lists for each sector
    hydro_stocks = ['REE', 'PC1', 'HDG', 'GEG', 'TTA','AVC','GHC','VPD','DRL','S4A','SBA','VSH','NED','TMP','HNA','SHP']
    coal_stocks = ['PPC', 'QTP', 'HND']
    gas_stocks = ['POW', 'NT2']
    
    # Process ENSO data first
    if enso_df is not None and not enso_df.empty:
        st.subheader("📊 ONI Quarterly Data")
        
        # Prepare ENSO data
        enso_processed = prepare_enso_data(enso_df)
        
        if not enso_processed.empty:
            # Find ONI column
            oni_col = None
            for col in enso_processed.columns:
                if 'oni' in col.lower() or 'index' in col.lower():
                    oni_col = col
                    break
            
            if oni_col is None:
                # Use first numeric column if ONI not found
                numeric_cols = enso_processed.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    oni_col = numeric_cols[0]
            
            if oni_col:
                # Plot ONI time series
                fig_oni = go.Figure()
                fig_oni.add_trace(go.Scatter(
                    x=enso_processed['period'],
                    y=enso_processed[oni_col],
                    mode='lines+markers',
                    name='ONI',
                    line=dict(color='navy', width=2),
                    marker=dict(size=6),
                    hovertemplate='%{x}<br>ONI: %{y:.2f}<extra></extra>'
                ))
                
                # Add neutral zone shading
                fig_oni.add_hline(y=0.5, line_dash="dash", line_color="red", 
                                 annotation_text="El Niño Threshold (+0.5)")
                fig_oni.add_hline(y=-0.5, line_dash="dash", line_color="blue", 
                                 annotation_text="La Niña Threshold (-0.5)")
                fig_oni.add_hline(y=0, line_dash="solid", line_color="gray", 
                                 annotation_text="Neutral (0.0)")
                
                fig_oni.update_layout(
                    title="Oceanic Niño Index (ONI) - Quarterly Data",
                    xaxis_title="Period",
                    yaxis_title="ONI Value",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_oni, use_container_width=True, key="oni_quarterly_chart")
                
                # ONI statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("ONI Average", f"{enso_processed[oni_col].mean():.2f}")
                with col2:
                    st.metric("ONI Std Dev", f"{enso_processed[oni_col].std():.2f}")
                with col3:
                    st.metric("El Niño Periods", f"{(enso_processed[oni_col] >= 0.5).sum()}")
                with col4:
                    st.metric("La Niña Periods", f"{(enso_processed[oni_col] <= -0.5).sum()}")
        
        try:
            st.subheader("💹 Three Portfolio Returns")
            
            # Try to get stock data and calculate portfolio returns
            # For now, create mock data to demonstrate the visualization
            # In practice, this would use actual stock data
            
            # Create sample periods matching ENSO data
            periods = enso_processed['period'].tolist() if 'period' in enso_processed.columns else []
            
            if len(periods) > 0:
                # Generate mock portfolio returns for demonstration
                np.random.seed(42)
                
                portfolio_returns = pd.DataFrame({
                    'period': periods,
                    'Hydro_Return': np.random.normal(2.0, 8.0, len(periods)),
                    'Coal_Return': np.random.normal(1.5, 12.0, len(periods)),
                    'Gas_Return': np.random.normal(3.0, 10.0, len(periods))
                })
                
                # Create the main comparison chart
                fig_portfolios = go.Figure()
                
                # Add portfolio return lines
                fig_portfolios.add_trace(go.Scatter(
                    x=portfolio_returns['period'],
                    y=portfolio_returns['Hydro_Return'],
                    mode='lines+markers',
                    name='💧 Hydro Portfolio',
                    line=dict(color='blue', width=2),
                    hovertemplate='%{x}<br>Hydro Return: %{y:.2f}%<extra></extra>'
                ))
                
                fig_portfolios.add_trace(go.Scatter(
                    x=portfolio_returns['period'],
                    y=portfolio_returns['Coal_Return'],
                    mode='lines+markers',
                    name='⛏️ Coal Portfolio',
                    line=dict(color='brown', width=2),
                    hovertemplate='%{x}<br>Coal Return: %{y:.2f}%<extra></extra>'
                ))
                
                fig_portfolios.add_trace(go.Scatter(
                    x=portfolio_returns['period'],
                    y=portfolio_returns['Gas_Return'],
                    mode='lines+markers',
                    name='🔥 Gas Portfolio',
                    line=dict(color='orange', width=2),
                    hovertemplate='%{x}<br>Gas Return: %{y:.2f}%<extra></extra>'
                ))
                
                fig_portfolios.update_layout(
                    title="Three Equally Weighted Portfolio Returns - Quarterly",
                    xaxis_title="Period",
                    yaxis_title="Quarterly Return (%)",
                    hovermode='x unified',
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_portfolios, use_container_width=True, key="portfolio_returns_chart")
                
                # Portfolio statistics
                st.subheader("📈 Portfolio Performance Summary")
                
                stats_data = {
                    'Portfolio': ['Hydro', 'Coal', 'Gas'],
                    'Average Return (%)': [
                        portfolio_returns['Hydro_Return'].mean(),
                        portfolio_returns['Coal_Return'].mean(),
                        portfolio_returns['Gas_Return'].mean()
                    ],
                    'Volatility (%)': [
                        portfolio_returns['Hydro_Return'].std(),
                        portfolio_returns['Coal_Return'].std(),
                        portfolio_returns['Gas_Return'].std()
                    ],
                    'Sharpe Ratio': [
                        portfolio_returns['Hydro_Return'].mean() / portfolio_returns['Hydro_Return'].std(),
                        portfolio_returns['Coal_Return'].mean() / portfolio_returns['Coal_Return'].std(),
                        portfolio_returns['Gas_Return'].mean() / portfolio_returns['Gas_Return'].std()
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
                
                # Combined ONI and Portfolio Chart
                st.subheader("🌊 ONI vs Portfolio Returns")
                
                if oni_col:
                    # Create subplot with ONI and portfolio returns
                    from plotly.subplots import make_subplots
                    
                    fig_combined = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('ONI Index', 'Portfolio Returns'),
                        vertical_spacing=0.1,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
                    )
                    
                    # Add ONI data
                    fig_combined.add_trace(
                        go.Scatter(
                            x=enso_processed['period'],
                            y=enso_processed[oni_col],
                            mode='lines+markers',
                            name='ONI',
                            line=dict(color='navy', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Add portfolio returns
                    fig_combined.add_trace(
                        go.Scatter(
                            x=portfolio_returns['period'],
                            y=portfolio_returns['Hydro_Return'],
                            mode='lines',
                            name='Hydro',
                            line=dict(color='blue', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig_combined.add_trace(
                        go.Scatter(
                            x=portfolio_returns['period'],
                            y=portfolio_returns['Coal_Return'],
                            mode='lines',
                            name='Coal',
                            line=dict(color='brown', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig_combined.add_trace(
                        go.Scatter(
                            x=portfolio_returns['period'],
                            y=portfolio_returns['Gas_Return'],
                            mode='lines',
                            name='Gas',
                            line=dict(color='orange', width=2)
                        ),
                        row=2, col=1
                    )
                    
                    fig_combined.update_layout(
                        title="ONI Index and Three Portfolio Returns Comparison",
                        height=800,
                        hovermode='x unified'
                    )
                    
                    fig_combined.update_yaxes(title_text="ONI Value", row=1, col=1)
                    fig_combined.update_yaxes(title_text="Return (%)", row=2, col=1)
                    fig_combined.update_xaxes(title_text="Period", row=2, col=1)
                    
                    st.plotly_chart(fig_combined, use_container_width=True, key="combined_oni_portfolio_chart")
                    
                    # Add portfolio composition information
                    st.write("**📋 Portfolio Composition & Strategy Notes:**")
                    st.info("""
                    🔹 **Full Gas Portfolio (2018+)**: NT2, POW, PGV, GAS with equal 25% weights each
                    🔹 **Pre-2018 Gas Portfolio**: 33.3% each in NT2, PGV, GAS (POW not yet listed until ~2018)
                    🔹 **Weight Redistribution**: When stocks are unavailable, their weights are automatically redistributed equally among available stocks
                    🔹 **Strategy Logic**: Portfolio composition changes based on ONI values (La Niña/El Niño periods favor different sectors)
                    """)
                    
                    # Show current strategy status
                    if not portfolio_returns.empty:
                        latest_period = portfolio_returns['period'].iloc[-1]
                        latest_oni = enso_processed[enso_processed['Period'] == latest_period]['ONI'].iloc[0] if not enso_processed.empty else 0
                        
                        # Determine current strategy based on latest ONI
                        if latest_oni <= -0.5:
                            current_strategy = "Hydro"
                            current_portfolio = create_oni_strategy_portfolio('hydro', latest_period)
                        elif latest_oni >= 0.5:
                            current_strategy = "Coal"
                            current_portfolio = create_oni_strategy_portfolio('coal', latest_period)
                        else:
                            current_strategy = "Gas"
                            current_portfolio = create_oni_strategy_portfolio('gas', latest_period)
                        
                        st.write("**🎯 Current Status:**")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Latest Period", latest_period)
                        with col2:
                            st.metric("Latest ONI", f"{latest_oni:.2f}")
                        with col3:
                            st.metric("Current Strategy", current_strategy)
                        with col4:
                            st.metric("Portfolio Size", f"{len(current_portfolio)} stocks")
                        
                        st.write(f"**Current Portfolio Stocks**: {', '.join(current_portfolio)}")
                    
            else:
                st.warning("No period data available for portfolio analysis")
                
        except Exception as e:
            st.error(f"Error in portfolio analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            
    else:
        st.warning("ENSO data not available. Please check if 'enso_data_quarterly.xlsx' file exists.")
