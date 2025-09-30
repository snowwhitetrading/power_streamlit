"""
Hydro Strategy Module
Implements flood level and flood capacity based stock selection strategy 
with quarterly portfolio rebalancing and performance analysis.

This module creates two portfolios:
1. Flood Level Portfolio - based on year-on-year growth in flood level from previous quarter
2. Flood Capacity Portfolio - based on year-on-year growth in flood capacity from previous quarter

Each portfolio contains:
- 1 liquid stock (most traded)
- 1 illiquid stock (less traded)

Performance is compared against:
- Equally weighted portfolio of all available stocks
- VNI Index cumulative return
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import warnings

# Try to import SSI API
try:
    import ssi_api
    SSI_API_AVAILABLE = True
except ImportError:
    SSI_API_AVAILABLE = False
    st.warning("SSI API not available. Using mock data for stock prices.")

warnings.filterwarnings('ignore')

def load_water_reservoir_data():
    """Load and process water reservoir data"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reservoir_file = os.path.join(script_dir, 'data', 'water_reservoir_monthly.csv')
        
        df = pd.read_csv(reservoir_file)
        
        # Convert date_time to datetime - the data appears to be in D/M/YYYY H:MM format
        try:
            df['date_time'] = pd.to_datetime(df['date_time'], format='%d/%m/%Y %H:%M')
        except ValueError as e:
            st.warning(f"Primary date format failed: {e}")
            # Fallback to flexible parsing with dayfirst=True for DD/MM/YYYY format
            try:
                df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True, errors='coerce')
            except:
                df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        
        # Remove rows with failed date parsing
        initial_count = len(df)
        df = df.dropna(subset=['date_time'])
        failed_count = initial_count - len(df)
        
        if failed_count > 0:
            st.warning(f"Could not parse {failed_count} date entries out of {initial_count} total records")
        
        # Validate we have some valid data
        if df.empty:
            st.error("No valid date entries found in water reservoir data")
            return pd.DataFrame()
        
        # Convert flood_level and flood_capacity to numeric
        df['flood_level'] = pd.to_numeric(df['flood_level'], errors='coerce')
        df['flood_capacity'] = pd.to_numeric(df['flood_capacity'], errors='coerce')
        
        # Filter out rows with zero or missing flood_capacity
        df = df[df['flood_capacity'] > 0]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading water reservoir data: {e}")
        return pd.DataFrame()

def load_stock_mappings():
    """Load stock mappings from water_list.csv"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lake_file = os.path.join(script_dir, 'data', 'water_list.csv')
        
        df = pd.read_csv(lake_file)
        
        # Create mappings dictionary
        mappings = {}
        liquid_stocks = set()
        illiquid_stocks = set()
        
        for _, row in df.iterrows():
            if len(row) < 4:
                continue
                
            region = row.iloc[0] if pd.notna(row.iloc[0]) else ""
            reservoir = row.iloc[1] if pd.notna(row.iloc[1]) else ""
            liquid = row.iloc[2] if pd.notna(row.iloc[2]) and str(row.iloc[2]).strip() != "" else None
            illiquid = row.iloc[3] if pd.notna(row.iloc[3]) and str(row.iloc[3]).strip() != "" else None
            
            if liquid:
                liquid_stocks.add(liquid.strip())
                # Initialize reservoir mapping if doesn't exist
                if reservoir not in mappings:
                    mappings[reservoir] = {'liquid': [], 'illiquid': []}
                # Handle multiple liquid stocks per reservoir
                if 'liquid' not in mappings[reservoir]:
                    mappings[reservoir]['liquid'] = []
                if isinstance(mappings[reservoir]['liquid'], str):
                    mappings[reservoir]['liquid'] = [mappings[reservoir]['liquid']]
                if liquid.strip() not in mappings[reservoir]['liquid']:
                    mappings[reservoir]['liquid'].append(liquid.strip())
                
            if illiquid:
                illiquid_stocks.add(illiquid.strip())
                # Initialize reservoir mapping if doesn't exist
                if reservoir not in mappings:
                    mappings[reservoir] = {'liquid': [], 'illiquid': []}
                # Handle multiple illiquid stocks per reservoir
                if 'illiquid' not in mappings[reservoir]:
                    mappings[reservoir]['illiquid'] = []
                if isinstance(mappings[reservoir]['illiquid'], str):
                    mappings[reservoir]['illiquid'] = [mappings[reservoir]['illiquid']]
                if illiquid.strip() not in mappings[reservoir]['illiquid']:
                    mappings[reservoir]['illiquid'].append(illiquid.strip())
        
        return mappings, list(liquid_stocks), list(illiquid_stocks)
        
    except Exception as e:
        st.error(f"Error loading stock mappings: {e}")
        return {}, [], []

def calculate_quarterly_growth_data(reservoir_df, mappings):
    """Calculate quarterly growth for flood level and capacity - QoQ for 2Q2020-1Q2021, YoY for rest"""
    try:
        # Add quarter column
        reservoir_df['quarter'] = reservoir_df['date_time'].dt.quarter
        reservoir_df['year'] = reservoir_df['date_time'].dt.year
        reservoir_df['period'] = reservoir_df.apply(lambda row: f"{int(row['year'])}Q{int(row['quarter'])}", axis=1)
        
        # Calculate quarterly averages by reservoir
        quarterly_data = reservoir_df.groupby(['period', 'reservoir_name']).agg({
            'flood_level': 'mean',
            'flood_capacity': 'mean'
        }).reset_index()
        
        # Calculate growth data (QoQ for 2Q2020-1Q2021, YoY for rest)
        growth_data = []
        
        for reservoir in quarterly_data['reservoir_name'].unique():
            reservoir_data = quarterly_data[quarterly_data['reservoir_name'] == reservoir].sort_values('period')
            
            # Skip if reservoir not in mappings
            if reservoir not in mappings:
                continue
                
            for metric in ['flood_level', 'flood_capacity']:
                for i in range(1, len(reservoir_data)):  # Start from 1st index
                    current_period = reservoir_data.iloc[i]['period']
                    current_value = reservoir_data.iloc[i][metric]
                    
                    # Determine growth type based on period
                    if current_period in ['2020Q2', '2020Q3', '2020Q4', '2021Q1']:
                        # Use quarter-on-quarter for 2Q2020-1Q2021 period
                        prev_value = reservoir_data.iloc[i-1][metric]
                        growth_col = 'qoq_growth'
                    else:
                        # Use year-on-year for the rest (from 2Q2021 onwards)
                        if i >= 4:  # Need at least 4 quarters for YoY
                            prev_value = reservoir_data.iloc[i-4][metric]
                            growth_col = 'yoy_growth'
                        else:
                            continue
                    
                    if pd.notna(current_value) and pd.notna(prev_value) and prev_value != 0:
                        growth = (current_value - prev_value) / prev_value * 100
                        
                        # Add data for both liquid and illiquid stocks if available
                        if 'liquid' in mappings[reservoir] and mappings[reservoir]['liquid']:
                            # Handle multiple liquid stocks for the same reservoir
                            liquid_stocks = mappings[reservoir]['liquid']
                            if isinstance(liquid_stocks, str):
                                liquid_stocks = [liquid_stocks]
                            
                            for liquid_stock in liquid_stocks:
                                record = {
                                    'period': current_period,
                                    'reservoir': reservoir,
                                    'stock': liquid_stock,
                                    'metric_type': metric,
                                    'stock_type': 'liquid'
                                }
                                record[growth_col] = growth
                                growth_data.append(record)
                            
                        if 'illiquid' in mappings[reservoir] and mappings[reservoir]['illiquid']:
                            # Handle multiple illiquid stocks for the same reservoir
                            illiquid_stocks = mappings[reservoir]['illiquid']
                            if isinstance(illiquid_stocks, str):
                                illiquid_stocks = [illiquid_stocks]
                            
                            for illiquid_stock in illiquid_stocks:
                                record = {
                                    'period': current_period,
                                    'reservoir': reservoir,
                                    'stock': illiquid_stock,
                                    'metric_type': metric,
                                    'stock_type': 'illiquid'
                                }
                                record[growth_col] = growth
                                growth_data.append(record)
        
        growth_df = pd.DataFrame(growth_data)
        
        # Aggregate growth by stock (simple average if stock has multiple reservoirs)
        if not growth_df.empty:
            aggregated_data = []
            
            for period in growth_df['period'].unique():
                period_data = growth_df[growth_df['period'] == period]
                
                for metric in ['flood_level', 'flood_capacity']:
                    metric_data = period_data[period_data['metric_type'] == metric]
                    
                    for stock_type in ['liquid', 'illiquid']:
                        stock_data = metric_data[metric_data['stock_type'] == stock_type]
                        
                        if not stock_data.empty:
                            # Group by stock and take average growth
                            for stock in stock_data['stock'].unique():
                                stock_records = stock_data[stock_data['stock'] == stock]
                                
                                # Determine growth column
                                if period in ['2020Q2', '2020Q3', '2020Q4', '2021Q1']:
                                    growth_col = 'qoq_growth'
                                else:
                                    growth_col = 'yoy_growth'
                                
                                if growth_col in stock_records.columns:
                                    avg_growth = stock_records[growth_col].mean()
                                    
                                    if pd.notna(avg_growth):
                                        aggregated_record = {
                                            'period': period,
                                            'stock': stock,
                                            'metric_type': metric,
                                            'stock_type': stock_type
                                        }
                                        aggregated_record[growth_col] = avg_growth
                                        aggregated_data.append(aggregated_record)
            
            return pd.DataFrame(aggregated_data)
        
        return growth_df
        
    except Exception as e:
        st.error(f"Error calculating quarterly growth: {e}")
        return pd.DataFrame()

def get_stock_data_ssi(stock_symbols):
    """Get stock price data using SSI API"""
    try:
        stock_data = {}
        
        if not SSI_API_AVAILABLE:
            st.error("❌ SSI API not available. Cannot fetch real stock data.")
            return {}
        
        end_date = datetime.now()
        start_date = datetime(2020, 1, 1)
        
        for symbol in stock_symbols:
            try:
                # Get data using SSI API
                data = ssi_api.fetch_historical_price(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if not data.empty:
                    # Ensure we have the required columns
                    if 'close' in data.columns and 'time' in data.columns:
                        data = data.rename(columns={'time': 'date'})
                        data['date'] = pd.to_datetime(data['date'])
                        stock_data[symbol] = data[['date', 'close']].copy()
                    else:
                        st.warning(f"Missing required columns for {symbol}")
                        
            except Exception as e:
                st.warning(f"Could not fetch data for {symbol}: {e}")
                continue
        
        if not stock_data:
            st.error("❌ Failed to fetch any stock data")
        
        return stock_data
        
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return {}

def convert_to_quarterly_returns(stock_data):
    """Convert daily stock data to quarterly returns"""
    try:
        quarterly_returns = {}
        
        for symbol, data in stock_data.items():
            if data is None or data.empty:
                continue
                
            # Add quarter columns
            data = data.copy()
            
            # Ensure date column is datetime
            data['date'] = pd.to_datetime(data['date'])
            
            data['year'] = data['date'].dt.year
            data['quarter'] = data['date'].dt.quarter
            
            # Create period string more safely
            data['period'] = data.apply(lambda row: f"{int(row['year'])}Q{int(row['quarter'])}", axis=1)
            
            # Get quarter-end prices
            quarter_data = data.groupby('period')['close'].last().reset_index()
            quarter_data = quarter_data.sort_values('period')
            
            # Calculate quarterly returns
            quarter_data['quarterly_return'] = quarter_data['close'].pct_change() * 100
            
            quarterly_returns[symbol] = quarter_data
        
        return quarterly_returns
        
    except Exception as e:
        st.error(f"Error converting to quarterly returns: {e}")
        return {}

def create_portfolios(growth_data, quarterly_returns):
    """Create flood level and flood capacity portfolios using corrected methodology:
    Use previous quarter's growth to select stocks for current quarter"""
    try:
        portfolios = {
            'flood_level': [],
            'flood_capacity': []
        }
        
        # Get available periods from 2Q2020 to 3Q2025
        # Start from 2020Q2 (baseline) - no stock selection needed for baseline
        periods = sorted(growth_data['period'].unique())
        target_periods = [p for p in periods if p >= '2020Q2' and p <= '2025Q3']
        
        # Helper function to get previous quarter
        def get_previous_quarter(period):
            year = int(period[:4])
            quarter = int(period[-1])
            
            if quarter == 1:
                return f"{year-1}Q4"
            else:
                return f"{year}Q{quarter-1}"
        
        for period in target_periods:
            # Handle baseline period (2020Q2) specially
            if period == '2020Q2':
                # Baseline period with 0% return
                for metric in ['flood_level', 'flood_capacity']:
                    portfolios[metric].append({
                        'period': period,
                        'selected_stocks': 'Baseline period',
                        'quarterly_return': 0.0,
                        'selection_based_on': 'N/A (baseline)'
                    })
                continue
                
            # Get previous quarter to use its growth for stock selection
            prev_period = get_previous_quarter(period)
            
            # Use PREVIOUS quarter's growth for selection (correct methodology)
            prev_growth = growth_data[growth_data['period'] == prev_period]
            
            # If no previous growth data, skip this period
            if prev_growth.empty:
                continue
            
            for metric in ['flood_level', 'flood_capacity']:
                metric_growth = prev_growth[prev_growth['metric_type'] == metric]
                
                if metric_growth.empty:
                    continue
                
                # Select best performing liquid and illiquid stocks from previous quarter
                liquid_data = metric_growth[metric_growth['stock_type'] == 'liquid']
                illiquid_data = metric_growth[metric_growth['stock_type'] == 'illiquid']
                
                selected_stocks = []
                portfolio_return = 0
                
                # Determine which growth column to use based on PREVIOUS period
                # This should match the growth calculation logic in calculate_quarterly_growth_data
                growth_col = 'qoq_growth' if prev_period in ['2020Q2', '2020Q3', '2020Q4', '2021Q1'] else 'yoy_growth'
                
                # Select best liquid stock from previous quarter (50% weight)
                if not liquid_data.empty and growth_col in liquid_data.columns:
                    valid_liquid = liquid_data.dropna(subset=[growth_col])
                    if not valid_liquid.empty:
                        best_liquid = valid_liquid.loc[valid_liquid[growth_col].idxmax()]
                        liquid_stock = best_liquid['stock']
                        selected_stocks.append(f"{liquid_stock} (L)")
                        
                        # Get stock return for CURRENT period (not previous)
                        if liquid_stock in quarterly_returns:
                            stock_data = quarterly_returns[liquid_stock]
                            period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                            if not period_return.empty:
                                portfolio_return += period_return.iloc[0] * 0.5  # 50% weight
                
                # Select best illiquid stock from previous quarter (50% weight)
                if not illiquid_data.empty and growth_col in illiquid_data.columns:
                    valid_illiquid = illiquid_data.dropna(subset=[growth_col])
                    if not valid_illiquid.empty:
                        best_illiquid = valid_illiquid.loc[valid_illiquid[growth_col].idxmax()]
                        illiquid_stock = best_illiquid['stock']
                        selected_stocks.append(f"{illiquid_stock} (I)")
                        
                        # Get stock return for CURRENT period (not previous)
                        if illiquid_stock in quarterly_returns:
                            stock_data = quarterly_returns[illiquid_stock]
                            period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                            if not period_return.empty:
                                portfolio_return += period_return.iloc[0] * 0.5  # 50% weight
                
                portfolios[metric].append({
                    'period': period,
                    'selected_stocks': ', '.join(selected_stocks),
                    'quarterly_return': portfolio_return,
                    'selection_based_on': prev_period  # Track which quarter's growth was used for selection
                })
        
        # Convert to DataFrames and calculate cumulative returns starting from 2020Q2 at 0%
        for metric in portfolios:
            df = pd.DataFrame(portfolios[metric])
            if not df.empty:
                # Add 2020Q2 as baseline with 0% return if not present
                if '2020Q2' not in df['period'].values:
                    baseline_row = pd.DataFrame({
                        'period': ['2020Q2'],
                        'selected_stocks': ['Baseline'],
                        'quarterly_return': [0],
                        'selection_based_on': ['Baseline']
                    })
                    df = pd.concat([baseline_row, df], ignore_index=True)
                
                # Ensure all quarters from 2020Q2 to current are represented
                all_expected_periods = []
                for year in range(2020, 2026):
                    for quarter in range(1, 5):
                        period = f"{year}Q{quarter}"
                        if period >= "2020Q2" and period <= "2025Q3":  # Start from 2020Q2
                            all_expected_periods.append(period)
                
                # Add missing periods with 0% return for continuity
                existing_periods = set(df['period'].values)
                for period in all_expected_periods:
                    if period not in existing_periods and period >= '2020Q2':
                        # Only add if we have some data after this period
                        later_periods = [p for p in existing_periods if p > period]
                        if later_periods:
                            missing_row = pd.DataFrame({
                                'period': [period],
                                'selected_stocks': ['No Data'],
                                'quarterly_return': [0],
                                'selection_based_on': ['No Data']
                            })
                            df = pd.concat([df, missing_row], ignore_index=True)
                
                df = df.sort_values('period').reset_index(drop=True)
                df['cumulative_return'] = (1 + df['quarterly_return'] / 100).cumprod() - 1
                df['cumulative_return'] *= 100
                portfolios[metric] = df
            else:
                portfolios[metric] = pd.DataFrame()
        
        return portfolios
        
    except Exception as e:
        st.error(f"Error creating portfolios: {e}")
        return {'flood_level': pd.DataFrame(), 'flood_capacity': pd.DataFrame()}

def get_previous_period(period):
    """Get previous quarter period"""
    year = int(period[:4])
    quarter = int(period[-1])
    
    if quarter == 1:
        return f"{year-1}Q4"
    else:
        return f"{year}Q{quarter-1}"

def create_benchmark_portfolios(all_stocks, quarterly_returns):
    """Create equally weighted portfolio and VNI index data starting from 1Q2020"""
    try:
        # Create equally weighted portfolio
        equal_weighted = []
        periods = set()
        
        # Get all available periods
        for stock_data in quarterly_returns.values():
            if not stock_data.empty:
                periods.update(stock_data['period'].tolist())
        
        periods = sorted([p for p in periods if p >= '2020Q2' and p <= '2025Q3'])
        
        for period in periods:
            total_return = 0
            stock_count = 0
            
            for stock in all_stocks:
                if stock in quarterly_returns:
                    stock_data = quarterly_returns[stock]
                    period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                    if not period_return.empty and pd.notna(period_return.iloc[0]):
                        total_return += period_return.iloc[0]
                        stock_count += 1
            
            if stock_count > 0:
                avg_return = total_return / stock_count
            else:
                avg_return = 0
                
            equal_weighted.append({
                'period': period,
                'quarterly_return': avg_return if period != '2020Q2' else 0  # Baseline for 2Q2020
            })
        
        equal_weighted_df = pd.DataFrame(equal_weighted)
        if not equal_weighted_df.empty:
            equal_weighted_df['cumulative_return'] = (1 + equal_weighted_df['quarterly_return'] / 100).cumprod() - 1
            equal_weighted_df['cumulative_return'] *= 100
        
        # Load VNI data
        vni_data = load_vni_data()
        
        return equal_weighted_df, vni_data
        
    except Exception as e:
        st.error(f"Error creating benchmark portfolios: {e}")
        return pd.DataFrame(), pd.DataFrame()

def load_vni_data():
    """Load VNI index data starting from 1Q2020"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vni_file = os.path.join(script_dir, 'data', 'vn_index_monthly.csv')
        
        if not os.path.exists(vni_file):
            return pd.DataFrame()
        
        df = pd.read_csv(vni_file)
        
        # Clean and rename columns
        if len(df.columns) >= 2:
            df.columns = ['period', 'vnindex_value'] + list(df.columns[2:])
            
            # Remove header rows like 'date', 'period', etc.
            df = df[~df['period'].astype(str).str.lower().isin(['date', 'period', 'time'])]
            
            # Convert vnindex_value to numeric, handle commas
            df['vnindex_value'] = df['vnindex_value'].astype(str).str.replace(',', '')
            df['vnindex_value'] = pd.to_numeric(df['vnindex_value'], errors='coerce')
            df = df.dropna(subset=['vnindex_value'])
            
            # Convert period format from "1Q2011" to "2011Q1" with better error handling
            def convert_period_format(period_str):
                try:
                    period_str = str(period_str).strip()
                    if 'Q' in period_str and len(period_str) > 3:
                        parts = period_str.split('Q')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            quarter = parts[0]
                            year = parts[1]
                            return f"{year}Q{quarter}"
                    return period_str
                except:
                    return period_str
                    
            df['period'] = df['period'].apply(convert_period_format)
            
            # Filter to only valid quarterly periods
            df = df[df['period'].str.contains(r'\d{4}Q\d', na=False)]
            
            # Filter for our period of interest (2Q2020 to 3Q2025)
            df = df[df['period'].between('2020Q2', '2025Q3')]
            
            # Calculate quarterly returns
            df = df.sort_values('period').reset_index(drop=True)
            df['quarterly_return'] = df['vnindex_value'].pct_change() * 100
            
            # Set 2Q2020 as baseline (0% return)
            df.loc[0, 'quarterly_return'] = 0
            
            df['cumulative_return'] = (1 + df['quarterly_return'] / 100).cumprod() - 1
            df['cumulative_return'] *= 100
            
            return df
        
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error loading VNI data: {e}")
        return pd.DataFrame()

def create_performance_chart(portfolios, equal_weighted_df, vni_data):
    """Create performance comparison chart"""
    try:
        fig = go.Figure()
        
        # Add flood level portfolio
        if not portfolios['flood_level'].empty:
            fig.add_trace(go.Scatter(
                x=portfolios['flood_level']['period'],
                y=portfolios['flood_level']['cumulative_return'],
                mode='lines+markers',
                name='🌊 Flood Level Portfolio',
                line=dict(color='#0C4130', width=3),
                hovertemplate='<b>Flood Level Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add flood capacity portfolio
        if not portfolios['flood_capacity'].empty:
            fig.add_trace(go.Scatter(
                x=portfolios['flood_capacity']['period'],
                y=portfolios['flood_capacity']['cumulative_return'],
                mode='lines+markers',
                name='💧 Flood Capacity Portfolio',
                line=dict(color='#08C179', width=3),
                hovertemplate='<b>Flood Capacity Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add equally weighted portfolio
        if not equal_weighted_df.empty:
            fig.add_trace(go.Scatter(
                x=equal_weighted_df['period'],
                y=equal_weighted_df['cumulative_return'],
                mode='lines+markers',
                name='⚖️ Equally Weighted Portfolio',
                line=dict(color='#D3BB96', width=2, dash='dash'),
                hovertemplate='<b>Equally Weighted Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add VNI Index
        if not vni_data.empty:
            fig.add_trace(go.Scatter(
                x=vni_data['period'],
                y=vni_data['cumulative_return'],
                mode='lines+markers',
                name='📈 VNI Index',
                line=dict(color='#97999B', width=2, dash='dot'),
                hovertemplate='<b>VNI Index</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Hydro Strategy Performance Comparison',
            xaxis_title='Period',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating performance chart: {e}")
        return go.Figure()

def create_performance_chart_filtered(portfolios, equal_weighted_df, vni_data, start_quarter, end_quarter):
    """Create performance comparison chart with time filtering"""
    try:
        fig = go.Figure()
        
        # Helper function to filter dataframe by time period
        def filter_by_period(df, start_q, end_q):
            if df.empty:
                return df
            return df[df['period'].between(start_q, end_q)]
        
        # Filter all dataframes
        flood_level_filtered = filter_by_period(portfolios['flood_level'], start_quarter, end_quarter)
        flood_capacity_filtered = filter_by_period(portfolios['flood_capacity'], start_quarter, end_quarter)
        equal_weighted_filtered = filter_by_period(equal_weighted_df, start_quarter, end_quarter)
        vni_filtered = filter_by_period(vni_data, start_quarter, end_quarter)
        
        # Recalculate cumulative returns from the starting point
        def recalculate_cumulative(df):
            if df.empty:
                return df
            df_copy = df.copy()
            # Reset cumulative return to start from 0 at the beginning of the period
            if len(df_copy) > 0:
                first_cumulative = df_copy['cumulative_return'].iloc[0]
                df_copy['cumulative_return'] = df_copy['cumulative_return'] - first_cumulative
            return df_copy
        
        flood_level_filtered = recalculate_cumulative(flood_level_filtered)
        flood_capacity_filtered = recalculate_cumulative(flood_capacity_filtered)
        equal_weighted_filtered = recalculate_cumulative(equal_weighted_filtered)
        vni_filtered = recalculate_cumulative(vni_filtered)
        
        # Add flood level portfolio
        if not flood_level_filtered.empty:
            fig.add_trace(go.Scatter(
                x=flood_level_filtered['period'],
                y=flood_level_filtered['cumulative_return'],
                mode='lines+markers',
                name='🌊 Flood Level Portfolio',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Flood Level Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add flood capacity portfolio
        if not flood_capacity_filtered.empty:
            fig.add_trace(go.Scatter(
                x=flood_capacity_filtered['period'],
                y=flood_capacity_filtered['cumulative_return'],
                mode='lines+markers',
                name='💧 Flood Capacity Portfolio',
                line=dict(color='green', width=3),
                hovertemplate='<b>Flood Capacity Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add equally weighted portfolio
        if not equal_weighted_filtered.empty:
            fig.add_trace(go.Scatter(
                x=equal_weighted_filtered['period'],
                y=equal_weighted_filtered['cumulative_return'],
                mode='lines+markers',
                name='⚖️ Equally Weighted Portfolio',
                line=dict(color='orange', width=2, dash='dash'),
                hovertemplate='<b>Equally Weighted Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add VNI Index
        if not vni_filtered.empty:
            fig.add_trace(go.Scatter(
                x=vni_filtered['period'],
                y=vni_filtered['cumulative_return'],
                mode='lines+markers',
                name='📈 VNI Index',
                line=dict(color='red', width=2, dash='dot'),
                hovertemplate='<b>VNI Index</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Portfolio Performance Comparison ({start_quarter} to {end_quarter})',
            xaxis_title='Period',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating filtered performance chart: {e}")
        return go.Figure()

def calculate_qoq_growth_data(reservoir_df, mappings, selected_quarter):
    """Calculate quarter-on-quarter growth for 2Q2020-1Q2021 period"""
    try:
        # Add quarter columns
        reservoir_df['quarter'] = reservoir_df['date_time'].dt.quarter
        reservoir_df['year'] = reservoir_df['date_time'].dt.year
        reservoir_df['period'] = reservoir_df.apply(lambda row: f"{int(row['year'])}Q{int(row['quarter'])}", axis=1)
        
        # Filter for the legacy period (2Q2020 to 1Q2021)
        legacy_periods = ['2020Q2', '2020Q3', '2020Q4', '2021Q1']
        reservoir_df = reservoir_df[reservoir_df['period'].isin(legacy_periods)]
        
        # Calculate quarterly averages by reservoir
        quarterly_data = reservoir_df.groupby(['period', 'reservoir_name']).agg({
            'flood_level': 'mean',
            'flood_capacity': 'mean'
        }).reset_index()
        
        # Calculate quarter-on-quarter growth
        growth_data = []
        
        for reservoir in quarterly_data['reservoir_name'].unique():
            reservoir_data = quarterly_data[quarterly_data['reservoir_name'] == reservoir].sort_values('period')
            
            # Skip if reservoir not in mappings
            if reservoir not in mappings:
                continue
                
            for metric in ['flood_level', 'flood_capacity']:
                for i in range(1, len(reservoir_data)):  # Start from 2nd quarter for QoQ
                    current_period = reservoir_data.iloc[i]['period']
                    current_value = reservoir_data.iloc[i][metric]
                    prev_value = reservoir_data.iloc[i-1][metric]
                    
                    if pd.notna(current_value) and pd.notna(prev_value) and prev_value != 0:
                        qoq_growth = (current_value - prev_value) / prev_value * 100
                        
                        # Add data for both liquid and illiquid stocks if available
                        if 'liquid' in mappings[reservoir] and mappings[reservoir]['liquid']:
                            # Handle multiple liquid stocks for the same reservoir
                            liquid_stocks = mappings[reservoir]['liquid']
                            if isinstance(liquid_stocks, str):
                                liquid_stocks = [liquid_stocks]
                            
                            for liquid_stock in liquid_stocks:
                                growth_data.append({
                                    'period': current_period,
                                    'reservoir': reservoir,
                                    'stock': liquid_stock,
                                    'metric_type': metric,
                                    'qoq_growth': qoq_growth,
                                    'stock_type': 'liquid'
                                })
                            
                        if 'illiquid' in mappings[reservoir] and mappings[reservoir]['illiquid']:
                            # Handle multiple illiquid stocks for the same reservoir
                            illiquid_stocks = mappings[reservoir]['illiquid']
                            if isinstance(illiquid_stocks, str):
                                illiquid_stocks = [illiquid_stocks]
                            
                            for illiquid_stock in illiquid_stocks:
                                growth_data.append({
                                    'period': current_period,
                                    'reservoir': reservoir,
                                    'stock': illiquid_stock,
                                    'metric_type': metric,
                                    'qoq_growth': qoq_growth,
                                    'stock_type': 'illiquid'
                                })
        
        return pd.DataFrame(growth_data)
        
    except Exception as e:
        st.error(f"Error calculating QoQ growth: {e}")
        return pd.DataFrame()

def create_qoq_portfolios(growth_data, quarterly_returns):
    """Create portfolios using QoQ growth for 2Q2020-1Q2021 period"""
    try:
        portfolios = {
            'flood_level': [],
            'flood_capacity': []
        }
        
        # Get available periods from the legacy period
        periods = sorted(growth_data['period'].unique())
        
        for period in periods:
            # For QoQ strategy, use current period's growth directly
            current_growth = growth_data[growth_data['period'] == period]
            
            if current_growth.empty:
                continue
            
            for metric in ['flood_level', 'flood_capacity']:
                metric_growth = current_growth[current_growth['metric_type'] == metric]
                
                if metric_growth.empty:
                    continue
                
                # Select best performing liquid and illiquid stocks
                liquid_data = metric_growth[metric_growth['stock_type'] == 'liquid']
                illiquid_data = metric_growth[metric_growth['stock_type'] == 'illiquid']
                
                selected_stocks = []
                portfolio_return = 0
                
                # Select best liquid stock (50% weight)
                if not liquid_data.empty:
                    best_liquid = liquid_data.loc[liquid_data['qoq_growth'].idxmax()]
                    liquid_stock = best_liquid['stock']
                    selected_stocks.append(f"{liquid_stock} (L)")
                    
                    # Get stock return for this period
                    if liquid_stock in quarterly_returns:
                        stock_data = quarterly_returns[liquid_stock]
                        period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                        if not period_return.empty:
                            portfolio_return += period_return.iloc[0] * 0.5  # 50% weight
                
                # Select best illiquid stock (50% weight)
                if not illiquid_data.empty:
                    best_illiquid = illiquid_data.loc[illiquid_data['qoq_growth'].idxmax()]
                    illiquid_stock = best_illiquid['stock']
                    selected_stocks.append(f"{illiquid_stock} (I)")
                    
                    # Get stock return for this period
                    if illiquid_stock in quarterly_returns:
                        stock_data = quarterly_returns[illiquid_stock]
                        period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                        if not period_return.empty:
                            portfolio_return += period_return.iloc[0] * 0.5  # 50% weight
                
                portfolios[metric].append({
                    'period': period,
                    'selected_stocks': ', '.join(selected_stocks),
                    'quarterly_return': portfolio_return
                })
        
        # Convert to DataFrames and calculate cumulative returns
        for metric in portfolios:
            df = pd.DataFrame(portfolios[metric])
            if not df.empty:
                df['cumulative_return'] = (1 + df['quarterly_return'] / 100).cumprod() - 1
                df['cumulative_return'] *= 100
                portfolios[metric] = df
            else:
                portfolios[metric] = pd.DataFrame()
        
        return portfolios
        
    except Exception as e:
        st.error(f"Error creating QoQ portfolios: {e}")
        return {'flood_level': pd.DataFrame(), 'flood_capacity': pd.DataFrame()}

def create_four_portfolio_chart(portfolios, equal_weighted_df, vni_data):
    """Create cumulative return chart for all 4 portfolios"""
    try:
        fig = go.Figure()
        
        # Add flood level portfolio
        if not portfolios['flood_level'].empty:
            fig.add_trace(go.Scatter(
                x=portfolios['flood_level']['period'],
                y=portfolios['flood_level']['cumulative_return'],
                mode='lines+markers',
                name='🌊 Flood Level Portfolio',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='<b>Flood Level Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add flood capacity portfolio
        if not portfolios['flood_capacity'].empty:
            fig.add_trace(go.Scatter(
                x=portfolios['flood_capacity']['period'],
                y=portfolios['flood_capacity']['cumulative_return'],
                mode='lines+markers',
                name='💧 Flood Capacity Portfolio',
                line=dict(color='#2ca02c', width=3),
                hovertemplate='<b>Flood Capacity Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add equally weighted portfolio
        if not equal_weighted_df.empty:
            fig.add_trace(go.Scatter(
                x=equal_weighted_df['period'],
                y=equal_weighted_df['cumulative_return'],
                mode='lines+markers',
                name='⚖️ Equally Weighted Portfolio',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='<b>Equally Weighted Portfolio</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        # Add VNI Index
        if not vni_data.empty:
            fig.add_trace(go.Scatter(
                x=vni_data['period'],
                y=vni_data['cumulative_return'],
                mode='lines+markers',
                name='📈 VNI Index',
                line=dict(color='#d62728', width=2, dash='dot'),
                hovertemplate='<b>VNI Index</b><br>' +
                            'Period: %{x}<br>' +
                            'Cumulative Return: %{y:.2f}%<br>' +
                            '<extra></extra>'
            ))
        
        fig.update_layout(
            title='Four Portfolio Cumulative Returns Comparison',
            xaxis_title='Period',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            template='plotly_white',
            height=600
        )
        
        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating four portfolio chart: {e}")
        return go.Figure()

def show_quarterly_growth_analysis(quarter_growth):
    """Show quarterly growth analysis for selected quarter"""
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌊 Flood Level Growth")
            flood_level_data = quarter_growth[quarter_growth['metric_type'] == 'flood_level']
            if not flood_level_data.empty:
                st.dataframe(flood_level_data[['stock', 'stock_type', 'reservoir', 'qoq_growth' if 'qoq_growth' in flood_level_data.columns else 'yoy_growth']])
                
                # Show top performers
                growth_col = 'qoq_growth' if 'qoq_growth' in flood_level_data.columns else 'yoy_growth'
                top_liquid = flood_level_data[flood_level_data['stock_type'] == 'liquid'].nlargest(1, growth_col)
                top_illiquid = flood_level_data[flood_level_data['stock_type'] == 'illiquid'].nlargest(1, growth_col)
                
                if not top_liquid.empty:
                    st.success(f"Top Liquid: {top_liquid.iloc[0]['stock']} ({top_liquid.iloc[0][growth_col]:.2f}%)")
                if not top_illiquid.empty:
                    st.success(f"Top Illiquid: {top_illiquid.iloc[0]['stock']} ({top_illiquid.iloc[0][growth_col]:.2f}%)")
            else:
                st.warning("No flood level data available")
        
        with col2:
            st.subheader("💧 Flood Capacity Growth")
            flood_capacity_data = quarter_growth[quarter_growth['metric_type'] == 'flood_capacity']
            if not flood_capacity_data.empty:
                st.dataframe(flood_capacity_data[['stock', 'stock_type', 'reservoir', 'qoq_growth' if 'qoq_growth' in flood_capacity_data.columns else 'yoy_growth']])
                
                # Show top performers
                growth_col = 'qoq_growth' if 'qoq_growth' in flood_capacity_data.columns else 'yoy_growth'
                top_liquid = flood_capacity_data[flood_capacity_data['stock_type'] == 'liquid'].nlargest(1, growth_col)
                top_illiquid = flood_capacity_data[flood_capacity_data['stock_type'] == 'illiquid'].nlargest(1, growth_col)
                
                if not top_liquid.empty:
                    st.success(f"Top Liquid: {top_liquid.iloc[0]['stock']} ({top_liquid.iloc[0][growth_col]:.2f}%)")
                if not top_illiquid.empty:
                    st.success(f"Top Illiquid: {top_illiquid.iloc[0]['stock']} ({top_illiquid.iloc[0][growth_col]:.2f}%)")
            else:
                st.warning("No flood capacity data available")
                
    except Exception as e:
        st.error(f"Error showing quarterly growth analysis: {e}")

def create_quarterly_growth_charts(growth_data, selected_quarter):
    """Create vertical bar charts for flood capacity and flood level growth"""
    try:
        quarter_data = growth_data[growth_data['period'] == selected_quarter]
        
        if quarter_data.empty:
            st.warning(f"No growth data available for {selected_quarter}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🌊 Flood Level Growth - {selected_quarter}")
            
            flood_level_data = quarter_data[quarter_data['metric_type'] == 'flood_level']
            if not flood_level_data.empty:
                # Determine which growth column to use
                growth_col = 'qoq_growth' if 'qoq_growth' in flood_level_data.columns and selected_quarter in ['2020Q2', '2020Q3', '2020Q4', '2021Q1'] else 'yoy_growth'
                
                # Check if growth column exists and has data
                if growth_col in flood_level_data.columns and not flood_level_data[growth_col].isna().all():
                    fig1 = go.Figure()
                    
                    # Add bars for liquid stocks
                    liquid_data = flood_level_data[flood_level_data['stock_type'] == 'liquid'].dropna(subset=[growth_col])
                    if not liquid_data.empty:
                        fig1.add_trace(go.Bar(
                            x=liquid_data['stock'],
                            y=liquid_data[growth_col],
                            name='Liquid Stocks',
                            marker_color='#08C179',
                            hovertemplate='<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>'
                        ))
                    
                    # Add bars for illiquid stocks
                    illiquid_data = flood_level_data[flood_level_data['stock_type'] == 'illiquid'].dropna(subset=[growth_col])
                    if not illiquid_data.empty:
                        fig1.add_trace(go.Bar(
                            x=illiquid_data['stock'],
                            y=illiquid_data[growth_col],
                            name='Illiquid Stocks',
                            marker_color='#0C4130',
                            hovertemplate='<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>'
                        ))
                    
                    growth_type_label = "QoQ" if growth_col == 'qoq_growth' else "YoY"
                    fig1.update_layout(
                        title=f'Flood Level {growth_type_label} Growth - {selected_quarter}',
                        xaxis_title='Stock Symbol',
                        yaxis_title='Growth (%)',
                        barmode='group',
                        template='plotly_white'
                    )
                    
                    fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig1, use_container_width=True)
                else:
                    st.warning(f"No growth data available for flood level in {selected_quarter}")
            else:
                st.warning("No flood level data available")
        
        with col2:
            st.subheader(f"💧 Flood Capacity Growth - {selected_quarter}")
            
            flood_capacity_data = quarter_data[quarter_data['metric_type'] == 'flood_capacity']
            if not flood_capacity_data.empty:
                # Determine which growth column to use
                growth_col = 'qoq_growth' if 'qoq_growth' in flood_capacity_data.columns and selected_quarter in ['2020Q2', '2020Q3', '2020Q4', '2021Q1'] else 'yoy_growth'
                
                # Check if growth column exists and has data
                if growth_col in flood_capacity_data.columns and not flood_capacity_data[growth_col].isna().all():
                    fig2 = go.Figure()
                    
                    # Add bars for liquid stocks
                    liquid_data = flood_capacity_data[flood_capacity_data['stock_type'] == 'liquid'].dropna(subset=[growth_col])
                    if not liquid_data.empty:
                        fig2.add_trace(go.Bar(
                            x=liquid_data['stock'],
                            y=liquid_data[growth_col],
                            name='Liquid Stocks',
                            marker_color='#08C179',
                            hovertemplate='<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>'
                        ))
                    
                    # Add bars for illiquid stocks
                    illiquid_data = flood_capacity_data[flood_capacity_data['stock_type'] == 'illiquid'].dropna(subset=[growth_col])
                    if not illiquid_data.empty:
                        fig2.add_trace(go.Bar(
                            x=illiquid_data['stock'],
                            y=illiquid_data[growth_col],
                            name='Illiquid Stocks',
                            marker_color='#0C4130',
                            hovertemplate='<b>%{x}</b><br>Growth: %{y:.2f}%<extra></extra>'
                        ))
                    
                    growth_type_label = "QoQ" if growth_col == 'qoq_growth' else "YoY"
                    fig2.update_layout(
                        title=f'Flood Capacity {growth_type_label} Growth - {selected_quarter}',
                        xaxis_title='Stock Symbol',
                        yaxis_title='Growth (%)',
                        barmode='group',
                        template='plotly_white'
                    )
                    
                    fig2.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning(f"No growth data available for flood capacity in {selected_quarter}")
            else:
                st.warning("No flood capacity data available")
                
    except Exception as e:
        st.error(f"Error creating quarterly growth charts: {e}")

def get_quarterly_returns(stock_mappings):
    """Get quarterly returns for all mapped stocks"""
    try:
        # Flatten mapped stocks into a simple list of ticker strings
        all_stocks = []
        for reservoir, stocks in stock_mappings.items():
            for key in ('liquid', 'illiquid'):
                if key in stocks:
                    val = stocks[key]
                    # Some reservoirs map to multiple stocks; ensure we flatten
                    if isinstance(val, list):
                        all_stocks.extend([s.strip() for s in val if isinstance(s, str) and s.strip()])
                    elif isinstance(val, str) and val.strip():
                        all_stocks.append(val.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for s in all_stocks:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        all_stocks = deduped
        
        # Get stock data
        stock_data = get_stock_data_ssi(all_stocks)
        
        # Convert to quarterly returns
        quarterly_returns = convert_to_quarterly_returns(stock_data)
        
        return quarterly_returns
        
    except Exception as e:
        st.error(f"Error getting quarterly returns: {e}")
        return {}

def run_flood_portfolio_strategy(strategy_type="Current (YoY Growth)", selected_quarter="2020Q2"):
    """Main function to run the Hydro strategy analysis"""
    try:
        st.markdown("""
        **Methodology**:
        - **Stock Selection**: Use previous quarter's year-on-year/quarter-on-quarter growth to select stocks for current quarter

        **🌊 Flood Level Portfolio**: Selects stocks based on reservoirs with highest flood level growth from previous quarter
                    
        **💧 Flood Capacity Portfolio**: Selects stocks based on reservoirs with highest flood capacity growth from previous quarter
        
        - **Time Periods (2Q2020 - 3Q2025)**: 
          - From 2Q2020 to 1Q2021: Uses quarter-on-quarter growth for stock selection
          - From 2Q2021 onwards: Uses year-on-year growth for stock selection

       **Portfolio**:
        - 1 liquid stock (50% weight) - REE PC1 HDG GEG
        - 1 illiquid stock (50% weight) - AVC GHC DRL HNA SHP CHP S4A SBA NED TMP VSH
        
        """)
        
        # Create tabs for better organization
        strategy_tab1, strategy_tab2, strategy_tab3 = st.tabs([
            "📊 Portfolio Performance", 
            "📈 Flood Data", 
            "📋 Portfolio Return"
        ])
        
        # Load data
        with st.spinner("Loading water reservoir data..."):
            reservoir_df = load_water_reservoir_data()
            
        if reservoir_df.empty:
            st.error("No water reservoir data available.")
            return
        
        with st.spinner("Loading stock mappings..."):
            mappings, liquid_stocks, illiquid_stocks = load_stock_mappings()
            
        if not mappings:
            st.error("No stock mappings available.")
            return
        
        all_stocks = list(set(liquid_stocks + illiquid_stocks))
        
        with st.spinner("Calculating quarterly growth data..."):
            growth_data = calculate_quarterly_growth_data(reservoir_df, mappings)
            
        if growth_data.empty:
            st.error("No growth data calculated.")
            return
        
        with st.spinner("Fetching stock price data..."):
            stock_data = get_stock_data_ssi(all_stocks)
            
        if not stock_data:
            st.error("No stock data available.")
            return
        
        with st.spinner("Converting to quarterly returns..."):
            quarterly_returns = convert_to_quarterly_returns(stock_data)
            
        with st.spinner("Creating portfolios..."):
            portfolios = create_portfolios(growth_data, quarterly_returns)
            
        with st.spinner("Creating benchmarks..."):
            equal_weighted_df, vni_data = create_benchmark_portfolios(all_stocks, quarterly_returns)
        
        # Tab 1: Portfolio Performance
        with strategy_tab1:
            st.subheader("📈 Performance Comparison")
            
            # Create and display performance chart (no time filtering, show all data)
            performance_chart = create_performance_chart(portfolios, equal_weighted_df, vni_data)
            st.plotly_chart(performance_chart, use_container_width=True)
        
        # Tab 2: Flood Data
        with strategy_tab2:
            st.subheader("📊 Flood Data Analysis")
            
            # Quarter selection for flood data
            available_quarters = sorted(growth_data['period'].unique())
            quarter_for_analysis = st.selectbox(
                "Select Quarter for Analysis:",
                available_quarters,
                index=len(available_quarters)//2 if available_quarters else 0,
                help="Choose a quarter to view flood capacity and level growth data"
            )
            
            # Show quarterly growth charts
            create_quarterly_growth_charts(growth_data, quarter_for_analysis)
        
        # Tab 3: Portfolio Return
        with strategy_tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🌊 Flood Level Portfolio Return")
                if not portfolios['flood_level'].empty:
                    st.dataframe(portfolios['flood_level'])
                    
                    # Show performance metrics
                    final_return = portfolios['flood_level']['cumulative_return'].iloc[-1] if not portfolios['flood_level'].empty else 0
                    st.metric("Final Cumulative Return", f"{final_return:.2f}%")
                else:
                    st.warning("No flood level portfolio data available")
            
            with col2:
                st.subheader("💧 Flood Capacity Portfolio Return")
                if not portfolios['flood_capacity'].empty:
                    st.dataframe(portfolios['flood_capacity'])
                    
                    # Show performance metrics
                    final_return = portfolios['flood_capacity']['cumulative_return'].iloc[-1] if not portfolios['flood_capacity'].empty else 0
                    st.metric("Final Cumulative Return", f"{final_return:.2f}%")
                else:
                    st.warning("No flood capacity portfolio data available")
        
        # Data quality info section removed as requested
        
    except Exception as e:
        st.error(f"Error running Hydro strategy: {e}")
        import traceback
        st.error(traceback.format_exc())

if __name__ == "__main__":
    run_flood_portfolio_strategy()
