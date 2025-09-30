"""
Coal Strategy Module
Implements coal volume based stock selection strategy with quarterly portfolio rebalancing.

This module creates two portfolios:
1. Diversified Portfolio - 50%/50% in two stocks with highest YoY volume growth from previous quarter
2. Concentrated Portfolio - 100% in single stock with highest YoY volume growth from previous quarter

Performance is compared against:
- Equally weighted portfolio of all coal stocks (PPC, QTP, HND)
- VN Index cumulative return

Coal companies: PPC, QTP, HND
Period: 1Q2019 to 3Q2025 (using 4Q2018 data for first selection)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings

# Try to import SSI API
try:
    from ssi_api import get_stock_data_batch
    SSI_API_AVAILABLE = True
except ImportError:
    SSI_API_AVAILABLE = False
    st.warning("SSI API not available. Using mock data for stock prices.")

def load_coal_volume_data():
    """Load and process coal volume quarterly data"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        coal_file = os.path.join(script_dir, 'data', 'coal_volume_quarterly.csv')
        
        df = pd.read_csv(coal_file)
        
        # The file format: first column is unnamed (period), then PPC, QTP, HND columns with long names
        # Rename columns to standard format
        if len(df.columns) == 4:
            df.columns = ['period', 'PPC', 'QTP', 'HND']
        else:
            # Try to handle different column structures
            st.error(f"Unexpected number of columns in coal data: {len(df.columns)}. Expected 4.")
            st.write("Available columns:", list(df.columns))
            return pd.DataFrame()
        
        # Remove any rows with empty periods
        df = df.dropna(subset=['period'])
        
        # Convert numeric columns
        for col in ['PPC', 'QTP', 'HND']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean period column and ensure proper format
        df['period'] = df['period'].astype(str).str.strip()
        
        # Filter out any invalid periods (should match pattern like "1Q2018", "4Q2017", etc.)
        df = df[df['period'].str.contains(r'\d+Q\d+', na=False)]
        
        # Convert period format from "1Q2018" to "2018Q1" to match other data
        def convert_period_format(period_str):
            try:
                if 'Q' in period_str:
                    parts = period_str.split('Q')
                    if len(parts) == 2:
                        quarter = parts[0]
                        year = parts[1]
                        return f"{year}Q{quarter}"
                return period_str
            except:
                return period_str
        
        df['period'] = df['period'].apply(convert_period_format)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading coal volume data: {e}")
        return pd.DataFrame()

def calculate_yoy_growth(coal_df):
    """Calculate year-on-year growth for coal volume data starting from 4Q2018"""
    try:
        coal_companies = ['PPC', 'QTP', 'HND']
        growth_data = []
        
        # Sort by period
        coal_df = coal_df.sort_values('period')
        
        # Calculate YoY growth for each company
        for company in coal_companies:
            if company not in coal_df.columns:
                continue
                
            company_data = coal_df[['period', company]].copy()
            company_data = company_data.sort_values('period')
            
            # Calculate YoY growth (4 quarters back)
            for i in range(4, len(company_data)):
                current_period = company_data.iloc[i]['period']
                current_value = company_data.iloc[i][company]
                prev_value = company_data.iloc[i-4][company]
                
                if pd.notna(current_value) and pd.notna(prev_value) and prev_value != 0:
                    yoy_growth = (current_value - prev_value) / prev_value * 100
                    
                    growth_data.append({
                        'period': current_period,
                        'company': company,
                        'volume': current_value,
                        'yoy_growth': yoy_growth
                    })
        
        growth_df = pd.DataFrame(growth_data)
        return growth_df
        
    except Exception as e:
        st.error(f"Error calculating YoY growth: {e}")
        return pd.DataFrame()

def fetch_stock_data(symbols):
    """Fetch stock price data using SSI API or generate mock data"""
    try:
        stock_data = {}
        
        if SSI_API_AVAILABLE:
            try:
                # Fetch real data from SSI API using the correct function name
                batch_data = get_stock_data_batch(symbols, '2018-01-01', '2025-06-30')
                
                for symbol in symbols:
                    if symbol in batch_data and not batch_data[symbol].empty:
                        data = batch_data[symbol].copy()
                        # Ensure the data has the expected structure
                        if 'close' in data.columns:
                            # Convert 'time' column to 'date' if it exists
                            if 'time' in data.columns:
                                data['date'] = pd.to_datetime(data['time'])
                            elif 'tradingDate' in data.columns:
                                data['date'] = pd.to_datetime(data['tradingDate'])
                            else:
                                st.warning(f"No date column found for {symbol}")
                                continue
                            
                            stock_data[symbol] = data
                        
                if stock_data:
                    return stock_data
                else:
                    st.error("❌ No data returned from SSI API")
                    return {}
                    
            except Exception as e:
                st.error(f"❌ Failed to fetch real stock data: {e}")
                return {}
        
        if not SSI_API_AVAILABLE:
            st.error("❌ SSI API not available. Cannot fetch real stock data.")
            return {}
        
        return stock_data
        
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return {}

def convert_to_quarterly_returns(stock_data):
    """Convert daily stock data to quarterly returns"""
    try:
        quarterly_returns = {}
        
        for symbol, data in stock_data.items():
            if data.empty:
                continue
                
            # Add quarter columns
            data = data.copy()
            data['year'] = data['date'].dt.year
            data['quarter'] = data['date'].dt.quarter
            data['period'] = data['year'].astype(str) + 'Q' + data['quarter'].astype(str)
            
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

def create_coal_portfolios(growth_data, quarterly_returns):
    """Create diversified and concentrated coal portfolios starting from 2018Q4 baseline"""
    try:
        portfolios = {
            'diversified': [],
            'concentrated': []
        }
        
        # Add 2018Q4 as baseline period with 0% return for both portfolios
        baseline_period = '2018Q4'
        portfolios['diversified'].append({
            'period': baseline_period,
            'stocks': 'Baseline',
            'quarterly_return': 0.0,
            'basis': 'Baseline (0%)'
        })
        
        portfolios['concentrated'].append({
            'period': baseline_period,
            'stocks': 'Baseline',
            'quarterly_return': 0.0,
            'basis': 'Baseline (0%)'
        })
        
        # Get available periods from 1Q2019 to 3Q2025
        # Start from 1Q2019 since we need 4Q2018 growth to select for 1Q2019
        periods = sorted(growth_data['period'].unique())
        target_periods = [p for p in periods if p >= '2019Q1' and p <= '2025Q2']
        
        # Helper function to get previous quarter
        def get_previous_quarter(period):
            year = int(period[:4])
            quarter = int(period[-1])
            
            if quarter == 1:
                return f"{year-1}Q4"
            else:
                return f"{year}Q{quarter-1}"
        
        for period in target_periods:
            # Get previous quarter to use its growth for stock selection
            prev_period = get_previous_quarter(period)
            
            # Get growth data for previous quarter
            prev_growth = growth_data[growth_data['period'] == prev_period]
            
            if prev_growth.empty:
                continue
            
            # Sort by YoY growth (descending)
            prev_growth = prev_growth.sort_values('yoy_growth', ascending=False)
            
            # DIVERSIFIED PORTFOLIO: Top 2 stocks with 50%/50% allocation
            diversified_return = 0
            diversified_stocks = []
            
            if len(prev_growth) >= 2:
                # Get top 2 stocks
                top_2_stocks = prev_growth.head(2)['company'].tolist()
                diversified_stocks = top_2_stocks
                
                # Calculate portfolio return (50%/50%)
                for stock in top_2_stocks:
                    if stock in quarterly_returns:
                        stock_data = quarterly_returns[stock]
                        period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                        if not period_return.empty:
                            diversified_return += period_return.iloc[0] * 0.5  # 50% weight
            
            # CONCENTRATED PORTFOLIO: Top 1 stock with 100% allocation
            concentrated_return = 0
            concentrated_stocks = []
            
            if len(prev_growth) >= 1:
                # Get top 1 stock
                top_stock = prev_growth.head(1)['company'].iloc[0]
                concentrated_stocks = [top_stock]
                
                # Calculate portfolio return (100%)
                if top_stock in quarterly_returns:
                    stock_data = quarterly_returns[top_stock]
                    period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                    if not period_return.empty:
                        concentrated_return = period_return.iloc[0]  # 100% weight
            
            # Store portfolio data
            portfolios['diversified'].append({
                'period': period,
                'selected_stocks': ', '.join(diversified_stocks),
                'quarterly_return': diversified_return,
                'selection_based_on': prev_period
            })
            
            portfolios['concentrated'].append({
                'period': period,
                'selected_stocks': ', '.join(concentrated_stocks),
                'quarterly_return': concentrated_return,
                'selection_based_on': prev_period
            })
        
        return portfolios
        
    except Exception as e:
        st.error(f"Error creating coal portfolios: {e}")
        return {'diversified': [], 'concentrated': []}

def create_benchmark_portfolios(quarterly_returns):
    """Create equally weighted portfolio of all coal stocks starting from 2018Q4 baseline"""
    try:
        coal_stocks = ['PPC', 'QTP', 'HND']
        benchmark_portfolio = []
        
        # Add 2018Q4 as baseline period with 0% return
        benchmark_portfolio.append({
            'period': '2018Q4',
            'quarterly_return': 0.0
        })
        
        # Get all available periods
        all_periods = set()
        for stock, data in quarterly_returns.items():
            if stock in coal_stocks:
                all_periods.update(data['period'].tolist())
        
        periods = sorted([p for p in all_periods if p >= '2019Q1' and p <= '2025Q2'])
        
        for period in periods:
            equal_weight_return = 0
            available_stocks = 0
            
            # Calculate equally weighted return
            for stock in coal_stocks:
                if stock in quarterly_returns:
                    stock_data = quarterly_returns[stock]
                    period_return = stock_data[stock_data['period'] == period]['quarterly_return']
                    if not period_return.empty:
                        equal_weight_return += period_return.iloc[0]
                        available_stocks += 1
            
            if available_stocks > 0:
                equal_weight_return = equal_weight_return / available_stocks
                
                benchmark_portfolio.append({
                    'period': period,
                    'quarterly_return': equal_weight_return
                })
        
        return benchmark_portfolio
        
    except Exception as e:
        st.error(f"Error creating benchmark portfolio: {e}")
        return []

def load_vni_data():
    """Load VNI data from CSV file and convert to quarterly returns"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vni_file = os.path.join(script_dir, 'vn_index_monthly.csv')
        
        vni_df = pd.read_csv(vni_file)
        
        # The VNI file structure: first column is period, second is VNINDEX value
        if len(vni_df.columns) >= 2:
            vni_df.columns = ['period', 'close'] + list(vni_df.columns[2:])
        else:
            st.error("VNI file does not have expected structure")
            return []
        
        # Clean and filter data - exclude header rows
        vni_df = vni_df.dropna(subset=['period', 'close'])
        
        # Remove header rows like 'date', 'period', etc.
        vni_df = vni_df[~vni_df['period'].astype(str).str.lower().isin(['date', 'period', 'time'])]
        
        # Convert close to numeric, handle commas
        vni_df['close'] = vni_df['close'].astype(str).str.replace(',', '')
        vni_df['close'] = pd.to_numeric(vni_df['close'], errors='coerce')
        vni_df = vni_df.dropna(subset=['close'])
        
        # Ensure period format is consistent and convert from "1Q2019" to "2019Q1" format
        vni_df['period'] = vni_df['period'].astype(str).str.strip()
        
        # Convert period format from "1Q2019" to "2019Q1" if needed
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
        
        vni_df['period'] = vni_df['period'].apply(convert_period_format)
        
        # Filter to only valid quarterly periods
        vni_df = vni_df[vni_df['period'].str.contains(r'\d{4}Q\d', na=False)]
        
        # Sort by period
        vni_df = vni_df.sort_values('period')
        
        # Calculate quarterly returns
        vni_df['quarterly_return'] = vni_df['close'].pct_change() * 100
        
        # Filter to target period (2019Q1 to 2025Q2)
        vni_filtered = vni_df[vni_df['period'].between('2019Q1', '2025Q2')]
        
        # Convert to list of dictionaries
        return vni_filtered[['period', 'quarterly_return']].dropna().to_dict('records')
        
    except Exception as e:
        st.error(f"Error loading VNI data: {e}")
        return []

def calculate_cumulative_returns(portfolio_data, start_period=None):
    """Calculate cumulative returns from quarterly returns, optionally starting from a specific period at 0%"""
    try:
        # Check if portfolio_data is empty
        if not portfolio_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(portfolio_data)
        
        # Check if required columns exist
        if df.empty:
            return pd.DataFrame()
            
        if 'period' not in df.columns:
            st.error("Portfolio data missing 'period' column")
            return pd.DataFrame()
            
        if 'quarterly_return' not in df.columns:
            st.error("Portfolio data missing 'quarterly_return' column") 
            return pd.DataFrame()
        
        # Sort by period
        df = df.sort_values('period')
        
        # If start_period is specified, filter data and set baseline
        if start_period:
            df = df[df['period'] >= start_period].copy()
            df = df.reset_index(drop=True)
            
            # Set the first period (start_period) return to 0% as baseline
            if not df.empty and df.iloc[0]['period'] == start_period:
                df.loc[0, 'quarterly_return'] = 0
        
        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['quarterly_return'] / 100).cumprod() - 1
        df['cumulative_return'] = df['cumulative_return'] * 100  # Convert to percentage
        
        return df
        
    except Exception as e:
        st.error(f"Error calculating cumulative returns: {e}")
        return pd.DataFrame()

def create_performance_chart(diversified_portfolio, concentrated_portfolio, benchmark_portfolio, vni_data):
    """Create performance comparison chart starting from 2018Q4 at 0%"""
    try:
        # Calculate cumulative returns starting from 2018Q4 at 0%
        diversified_df = calculate_cumulative_returns(diversified_portfolio, start_period='2018Q4')
        concentrated_df = calculate_cumulative_returns(concentrated_portfolio, start_period='2018Q4')
        benchmark_df = calculate_cumulative_returns(benchmark_portfolio, start_period='2018Q4')
        vni_df = pd.DataFrame(vni_data)
        vni_df = vni_df.sort_values('period')
        
        # Filter VNI data to start from 2018Q4 and set as baseline
        vni_df = vni_df[vni_df['period'] >= '2018Q4'].copy()
        vni_df = vni_df.reset_index(drop=True)
        
        # Calculate cumulative returns for VNI starting from 2018Q4 at 0%
        if not vni_df.empty:
            # Set 2018Q4 as baseline (0%)
            if vni_df.iloc[0]['period'] == '2018Q4':
                vni_df.loc[0, 'quarterly_return'] = 0
            
            vni_df['cumulative_return'] = (1 + vni_df['quarterly_return'] / 100).cumprod() - 1
            vni_df['cumulative_return'] = vni_df['cumulative_return'] * 100
        
        # Create the chart
        fig = go.Figure()
        
        # Add diversified portfolio line
        fig.add_trace(go.Scatter(
            x=diversified_df['period'],
            y=diversified_df['cumulative_return'],
            mode='lines+markers',
            name='Diversified Portfolio (50%/50%)',
            line=dict(color='#0C4130', width=3),
            marker=dict(size=6),
            hovertemplate="<b>Diversified Portfolio</b><br>" +
                         "Period: %{x}<br>" +
                         "Cumulative Return: %{y:.2f}%<br>" +
                         "<extra></extra>"
        ))
        
        # Add concentrated portfolio line
        fig.add_trace(go.Scatter(
            x=concentrated_df['period'],
            y=concentrated_df['cumulative_return'],
            mode='lines+markers',
            name='Concentrated Portfolio (100%)',
            line=dict(color='#08C179', width=3),
            marker=dict(size=6),
            hovertemplate="<b>Concentrated Portfolio</b><br>" +
                         "Period: %{x}<br>" +
                         "Cumulative Return: %{y:.2f}%<br>" +
                         "<extra></extra>"
        ))
        
        # Add benchmark portfolio line with dash style
        fig.add_trace(go.Scatter(
            x=benchmark_df['period'],
            y=benchmark_df['cumulative_return'],
            mode='lines+markers',
            name='Equally Weighted Portfolio',
            line=dict(color='#B78D51', width=3, dash='dash'),
            marker=dict(size=6),
            hovertemplate="<b>Equally Weighted Portfolio</b><br>" +
                         "Period: %{x}<br>" +
                         "Cumulative Return: %{y:.2f}%<br>" +
                         "<extra></extra>"
        ))
        
        # Add VNI line with dash style
        fig.add_trace(go.Scatter(
            x=vni_df['period'],
            y=vni_df['cumulative_return'],
            mode='lines+markers',
            name='VN Index',
            line=dict(color='#97999B', width=3, dash='dash'),
            marker=dict(size=6),
            hovertemplate="<b>VN Index</b><br>" +
                         "Period: %{x}<br>" +
                         "Cumulative Return: %{y:.2f}%<br>" +
                         "<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Coal Strategy - Portfolio Performance Comparison (Starting from 2018Q4 at 0%)",
            xaxis_title="Quarter",
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
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating performance chart: {e}")
        return go.Figure()

def create_portfolio_details_table(portfolio_data, portfolio_name):
    """Create a table showing portfolio details"""
    try:
        # Check if portfolio_data is empty
        if not portfolio_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(portfolio_data)
        
        # Check if required columns exist
        if df.empty:
            return pd.DataFrame()
            
        if 'period' not in df.columns:
            st.error(f"Portfolio data for {portfolio_name} missing 'period' column")
            return pd.DataFrame()
            
        if 'quarterly_return' not in df.columns:
            st.error(f"Portfolio data for {portfolio_name} missing 'quarterly_return' column") 
            return pd.DataFrame()
        
        df = df.sort_values('period')
        
        # Calculate cumulative returns
        df['cumulative_return'] = (1 + df['quarterly_return'] / 100).cumprod() - 1
        df['cumulative_return'] = df['cumulative_return'] * 100
        
        # Format for display
        df_display = df.copy()
        df_display['quarterly_return'] = df_display['quarterly_return'].round(4)
        df_display['cumulative_return'] = df_display['cumulative_return'].round(4)
        
        return df_display
        
    except Exception as e:
        st.error(f"Error creating portfolio details table: {e}")
        return pd.DataFrame()

def run_coal_strategy():
    """Main function to run the coal strategy analysis"""
    try:
        st.header("Coal Strategy Analysis")
        st.markdown("""
        **Methodology:**
        - **Diversified Portfolio**: 50%/50% allocation to two stocks with highest YoY volume growth from previous quarter
        - **Concentrated Portfolio**: 100% allocation to one stock with highest YoY volume growth from previous quarter
        - **Time Period**: 1Q2019 to 3Q2025
        
        **Portfolio**: PPC, QTP, HND
        """)
        
        # Load coal volume data
        coal_df = load_coal_volume_data()
        if coal_df.empty:
            st.error("Could not load coal volume data")
            return
        
        # Calculate YoY growth
        growth_data = calculate_yoy_growth(coal_df)
        if growth_data.empty:
            st.error("Could not calculate YoY growth data")
            return
        
        # Fetch stock price data
        coal_stocks = ['PPC', 'QTP', 'HND']
        stock_data = fetch_stock_data(coal_stocks)
        if not stock_data:
            st.error("Could not fetch stock price data")
            return
        
        # Convert to quarterly returns
        quarterly_returns = convert_to_quarterly_returns(stock_data)
        
        # Create portfolios
        portfolios = create_coal_portfolios(growth_data, quarterly_returns)
        benchmark_portfolio = create_benchmark_portfolios(quarterly_returns)
        vni_data = load_vni_data()
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Portfolio Performance", "📊 Portfolio Return", "📋 Volume Growth"])
        
        with tab1:
            st.subheader("Portfolio Performance Comparison")
            
            # Performance chart
            performance_fig = create_performance_chart(
                portfolios['diversified'],
                portfolios['concentrated'],
                benchmark_portfolio,
                vni_data
            )
            st.plotly_chart(performance_fig, use_container_width=True)
            
            # Final performance summary
            col1, col2, col3, col4 = st.columns(4)
            
            try:
                diversified_final = calculate_cumulative_returns(portfolios['diversified'])['cumulative_return'].iloc[-1]
                col1.metric("Diversified Portfolio", f"{diversified_final:.2f}%")
            except:
                col1.metric("Diversified Portfolio", "N/A")
            
            try:
                concentrated_final = calculate_cumulative_returns(portfolios['concentrated'])['cumulative_return'].iloc[-1]
                col2.metric("Concentrated Portfolio", f"{concentrated_final:.2f}%")
            except:
                col2.metric("Concentrated Portfolio", "N/A")
            
            try:
                benchmark_final = calculate_cumulative_returns(benchmark_portfolio)['cumulative_return'].iloc[-1]
                col3.metric("Equally Weighted", f"{benchmark_final:.2f}%")
            except:
                col3.metric("Equally Weighted", "N/A")
            
            try:
                vni_df = pd.DataFrame(vni_data)
                vni_df['cumulative_return'] = (1 + vni_df['quarterly_return'] / 100).cumprod() - 1
                vni_final = vni_df['cumulative_return'].iloc[-1] * 100
                col4.metric("VN Index", f"{vni_final:.2f}%")
            except:
                col4.metric("VN Index", "N/A")
        
        with tab2:
            st.subheader("Portfolio Details")
            
            # Portfolio details tables
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Diversified Portfolio Details**")
                diversified_details = create_portfolio_details_table(portfolios['diversified'], "Diversified")
                st.dataframe(diversified_details, use_container_width=True)
            
            with col2:
                st.write("**Concentrated Portfolio Details**")
                concentrated_details = create_portfolio_details_table(portfolios['concentrated'], "Concentrated")
                st.dataframe(concentrated_details, use_container_width=True)
        
        with tab3:
            st.subheader("Coal Volume Growth Data")
            
            # Growth data table
            st.write("**Year-on-Year Volume Growth by Quarter**")
            growth_pivot = growth_data.pivot(index='period', columns='company', values='yoy_growth')
            st.dataframe(growth_pivot, use_container_width=True)
            
            # Volume growth chart
            fig_growth = go.Figure()
            
            colors = {'PPC': '#0C4130', 'QTP': '#08C179', 'HND': '#B78D51'}
            
            for company in ['PPC', 'QTP', 'HND']:
                company_data = growth_data[growth_data['company'] == company]
                if not company_data.empty:
                    fig_growth.add_trace(go.Scatter(
                        x=company_data['period'],
                        y=company_data['yoy_growth'],
                        mode='lines+markers',
                        name=f'{company} YoY Growth',
                        line=dict(color=colors.get(company, '#000000'), width=2),
                        marker=dict(size=6)
                    ))
            
            fig_growth.update_layout(
                title="Coal Companies - Year-on-Year Volume Growth",
                xaxis_title="Quarter",
                yaxis_title="YoY Growth (%)",
                height=500,
                hovermode='x unified'
            )
            
            fig_growth.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            st.plotly_chart(fig_growth, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error running coal strategy: {e}")

if __name__ == "__main__":
    run_coal_strategy()
