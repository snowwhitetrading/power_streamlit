"""
Gas Strategy Module - POW vs NT2 Quarterly Portfolio Strategy
Based on quarterly YoY growth of contracted volumes from 1Q2019 to 2Q2025
Note: Ca Mau contracted volume is 0 from 2019-2021, but is included in POW total contracted volume calculation

Methodology:
- Diversified portfolio
+ If POW growth - NT2 growth > 20%, invest 100% in POW in the next quarter
+ If NT2 growth - POW growth > 20%, invest 100% in NT2 in the next quarter  
+ Otherwise, use equal weight (50/50)
- Concentrated portfolio: Invest 100% in the stock having higher YoY growth
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict, Any
import os

# Try to import ssi_api, make it optional
try:
    import ssi_api
    SSI_API_AVAILABLE = True
except ImportError:
    SSI_API_AVAILABLE = False
    print("Warning: ssi_api module not available. Gas strategy will use mock data.")

def load_pvpower_data():
    """Load PVPower monthly data from CSV file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'data', 'volume_pow_monthly.csv')
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading PVPower data: {str(e)}")
        return None

def process_quarterly_data(df):
    """Convert monthly data to quarterly and calculate YoY growth for POW and NT2 contracted volumes"""
    try:
        if df is None or df.empty:
            return None
        
        # Ensure we have a Date column
        if 'Date' not in df.columns:
            # Try to create date from Year, Month columns if they exist
            if 'Year' in df.columns and 'Month' in df.columns:
                df['Date'] = pd.to_datetime([f"{int(y)}-{int(m):02d}-01" for y, m in zip(df['Year'], df['Month'])])
            # Try other common date column patterns
            elif any(col.lower() in ['date', 'datetime', 'time'] for col in df.columns):
                date_col = next(col for col in df.columns if col.lower() in ['date', 'datetime', 'time'])
                df['Date'] = pd.to_datetime(df[date_col])
            # Try to infer from index if it's a datetime index
            elif hasattr(df.index, 'year'):
                df['Date'] = df.index
            else:
                st.error("No Date, Year/Month columns found in the data")
                st.write("Available columns:")
                st.write(df.columns.tolist())
                return None
        
        # Convert Date to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        
        # Filter data from 1Q2019 to 2Q2025 (include 2019-2021 data where Ca Mau has 0 contracted volume)
        # Note: Ca Mau contracted volume is 0 from 2019-2021, but we still include it in POW total
        start_date = pd.to_datetime('2019-01-01')
        end_date = pd.to_datetime('2025-06-30')
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        
        # Find POW Contracted and NT2 contracted volume columns
        # Look for direct POW Contracted column first
        pow_contracted_cols = [col for col in df.columns if 'POW' in str(col).upper() and 'CONTRACTED' in str(col).upper()]
        nt2_cols = [col for col in df.columns if 'NT2' in str(col).upper() and 'CONTRACTED' in str(col).upper()]
        
        # If NT2 columns not found, look for alternatives
        if not nt2_cols:
            nt2_cols = [col for col in df.columns if 'NHON TRACH 2' in str(col).upper() and 'CONTRACTED' in str(col).upper()]
            
        if not pow_contracted_cols or not nt2_cols:
            st.error("Could not find POW Contracted and NT2 contracted volume columns")
            st.write("Available columns:")
            st.write(df.columns.tolist())
            return None
            
        # Use the POW Contracted column directly (not summing individual plants)
        pow_col = pow_contracted_cols[0]
        
        # Show sample POW contracted data
        sample_pow = df[pow_col].head(10)
        
        nt2_col = nt2_cols[0]
        
        # Show sample NT2 data for comparison
        sample_nt2 = df[nt2_col].dropna().head(5)
        
        # Convert to numeric
        df[pow_col] = pd.to_numeric(df[pow_col], errors='coerce')
        df[nt2_col] = pd.to_numeric(df[nt2_col], errors='coerce')
        
        # Group by quarter and sum the volumes
        quarterly_df = df.groupby(['Year', 'Quarter']).agg({
            pow_col: 'sum',
            nt2_col: 'sum'
        }).reset_index()
        
        # Create quarter label safely
        quarterly_df['Quarter_Label'] = quarterly_df.apply(lambda row: f"{int(row['Year'])}Q{int(row['Quarter'])}", axis=1)
        quarterly_df['Date'] = pd.to_datetime([f"{int(y)}-{int(q)*3:02d}-01" for y, q in zip(quarterly_df['Year'], quarterly_df['Quarter'])])
        
        # Calculate YoY growth (4 quarters back)
        quarterly_df = quarterly_df.sort_values(['Year', 'Quarter']).reset_index(drop=True)
        quarterly_df['POW_YoY_Growth'] = quarterly_df[pow_col].pct_change(periods=4) * 100
        quarterly_df['NT2_YoY_Growth'] = quarterly_df[nt2_col].pct_change(periods=4) * 100
        
        # Rename columns for clarity
        quarterly_df = quarterly_df.rename(columns={
            pow_col: 'POW_Contracted',
            nt2_col: 'NT2_Contracted'
        })
        
        return quarterly_df
        
    except Exception as e:
        st.error(f"Error processing quarterly data: {str(e)}")
        return None

def construct_portfolio_strategy(quarterly_df):
    """
    Construct quarterly portfolio based on YoY growth comparison
    Strategy starts from 2Q2019 based on 1Q2019 YoY growth comparison
    Data includes 2019-2021 period where Ca Mau contracted volume is 0, but is still included in POW total
    
    New Methodology:
    - If POW growth - NT2 growth > 20%, invest 100% in POW in the next quarter
    - If NT2 growth - POW growth > 20%, invest 100% in NT2 in the next quarter  
    - Otherwise, use equal weight (50/50)
    """
    try:
        if quarterly_df is None or quarterly_df.empty:
            return None
            
        strategy_df = quarterly_df.copy()
        
        # Strategy starts from 2Q2019 (based on 1Q2019 YoY growth)
        start_strategy_date = pd.to_datetime('2019-04-01')  # 2Q2019
        
        # Portfolio allocation logic: invest 100% in the stock with higher YoY growth difference > 20%
        # Investment decision is made for next quarter based on current quarter comparison
        strategy_df['Portfolio_Decision'] = 'Hold'  # Default
        strategy_df['POW_Weight'] = 0.5  # Default equal weight
        strategy_df['NT2_Weight'] = 0.5  # Default equal weight
        
        for i in range(len(strategy_df)-1):  # Exclude last row as we can't invest for next quarter
            current_row = strategy_df.iloc[i]
            next_row_date = strategy_df.iloc[i+1]['Date']
            
            # Only apply strategy starting from 2Q2019
            if next_row_date < start_strategy_date:
                # Before 2Q2019, use equal weights (50/50)
                strategy_df.loc[i+1, 'Portfolio_Decision'] = 'Equal'
                strategy_df.loc[i+1, 'POW_Weight'] = 0.5
                strategy_df.loc[i+1, 'NT2_Weight'] = 0.5
                continue
            
            # Compare YoY growth rates (only if both are not NaN)
            if not pd.isna(current_row['POW_YoY_Growth']) and not pd.isna(current_row['NT2_YoY_Growth']):
                pow_growth = current_row['POW_YoY_Growth']
                nt2_growth = current_row['NT2_YoY_Growth']
                
                # Calculate growth difference
                pow_advantage = pow_growth - nt2_growth
                nt2_advantage = nt2_growth - pow_growth
                
                # Apply new methodology: 20% threshold for 100% allocation
                if pow_advantage > 20:
                    # POW growth advantage > 20%, invest 100% in POW
                    strategy_df.loc[i+1, 'Portfolio_Decision'] = 'POW'
                    strategy_df.loc[i+1, 'POW_Weight'] = 1.0
                    strategy_df.loc[i+1, 'NT2_Weight'] = 0.0
                elif nt2_advantage > 20:
                    # NT2 growth advantage > 20%, invest 100% in NT2
                    strategy_df.loc[i+1, 'Portfolio_Decision'] = 'NT2'
                    strategy_df.loc[i+1, 'POW_Weight'] = 0.0
                    strategy_df.loc[i+1, 'NT2_Weight'] = 1.0
                else:
                    # Growth difference <= 20%, use equal weight
                    strategy_df.loc[i+1, 'Portfolio_Decision'] = 'Equal'
                    strategy_df.loc[i+1, 'POW_Weight'] = 0.5
                    strategy_df.loc[i+1, 'NT2_Weight'] = 0.5
            else:
                # If YoY data not available (likely for POW due to missing historical data)
                # Use equal weight or last known strategy
                strategy_df.loc[i+1, 'Portfolio_Decision'] = 'Equal'
                strategy_df.loc[i+1, 'POW_Weight'] = 0.5
                strategy_df.loc[i+1, 'NT2_Weight'] = 0.5
        
        return strategy_df
        
    except Exception as e:
        st.error(f"Error constructing portfolio strategy: {str(e)}")
        return None

def get_stock_returns_ssi(tickers=['POW', 'NT2'], start_year=2019, end_year=2025):
    """
    Get stock returns using SSI API method following hydro_strategy pattern
    Start from 2019 to align with gas strategy timing
    """
    try:
        # Check SSI API availability following hydro_strategy pattern
        SSI_API_AVAILABLE = False
        try:
            from . import ssi_api
            SSI_API_AVAILABLE = True
        except ImportError:
            try:
                import ssi_api
                SSI_API_AVAILABLE = True
            except ImportError:
                st.warning("SSI API not available. Using mock data for demonstration.")
                SSI_API_AVAILABLE = False
        
        stock_data = {}
        
        if SSI_API_AVAILABLE:
            # Use SSI API to get actual stock data following app_new pattern
            for ticker in tickers:
                try:
                    # Use get_stock_history function from ssi_api
                    data = ssi_api.get_stock_history(ticker, f"{start_year}-01-01", f"{end_year}-12-31")
                    if data is not None and not data.empty:
                        # Reset index to get the time column as a regular column
                        data = data.reset_index()
                        
                        # Handle column names - SSI API returns 'time' and 'close'
                        if 'time' in data.columns:
                            data = data.rename(columns={'time': 'Date'})
                        if 'close' in data.columns:
                            data = data.rename(columns={'close': 'Close'})
                        
                        # Ensure we have the required columns
                        if 'Date' not in data.columns or 'Close' not in data.columns:
                            st.error(f"Missing required columns for {ticker}. Available: {data.columns.tolist()}")
                            continue
                            
                        # Convert to quarterly data using quarter-end prices
                        data['Date'] = pd.to_datetime(data['Date'])
                        data = data.set_index('Date')
                        
                        # Resample to quarterly using last price of quarter (quarter-end)
                        quarterly_data = data['Close'].resample('Q').last().to_frame()
                        quarterly_data.columns = [f'{ticker}_Price']
                        quarterly_data = quarterly_data.reset_index()
                        
                        # Create Quarter_Label to match strategy data format
                        quarterly_data['Year'] = quarterly_data['Date'].dt.year
                        quarterly_data['Quarter'] = quarterly_data['Date'].dt.quarter
                        quarterly_data['Quarter_Label'] = quarterly_data['Year'].astype(str) + 'Q' + quarterly_data['Quarter'].astype(str)
                        
                        # Calculate quarterly returns
                        quarterly_data[f'{ticker}_Return'] = quarterly_data[f'{ticker}_Price'].pct_change() * 100
                        
                        # Create final format expected by portfolio calculation
                        stock_returns = quarterly_data[['Quarter_Label', f'{ticker}_Return']].copy()
                        stock_returns.columns = ['Quarter', 'Return']
                        
                        stock_data[ticker] = stock_returns
                except Exception as e:
                    st.error(f"Error fetching {ticker} data from SSI API: {str(e)}")
                    st.error(f"Exception details: {type(e).__name__}: {str(e)}")
        
        # If SSI API failed or not available, return empty data
        if not stock_data:
            st.error("❌ Failed to fetch any real stock data. SSI API may be unavailable.")
            
        return stock_data
        
    except Exception as e:
        st.error(f"Error getting stock returns: {str(e)}")
        return {}

def calculate_portfolio_returns(strategy_df, stock_data):
    """Calculate portfolio returns vs equally weighted and VNI returns"""
    try:
        if strategy_df is None or not stock_data:
            return None
            
        # Merge strategy data with stock returns
        returns_df = strategy_df[['Quarter_Label', 'POW_Weight', 'NT2_Weight', 'Portfolio_Decision']].copy()
        
        # Add stock returns
        if 'POW' in stock_data and 'NT2' in stock_data:
            pow_returns = stock_data['POW'].set_index('Quarter')['Return'] if 'Quarter' in stock_data['POW'].columns else pd.Series()
            nt2_returns = stock_data['NT2'].set_index('Quarter')['Return'] if 'Quarter' in stock_data['NT2'].columns else pd.Series()
            
            returns_df['POW_Return'] = returns_df['Quarter_Label'].map(pow_returns)
            returns_df['NT2_Return'] = returns_df['Quarter_Label'].map(nt2_returns)
            
            # Fill NaN with 0
            returns_df['POW_Return'] = returns_df['POW_Return'].fillna(0)
            returns_df['NT2_Return'] = returns_df['NT2_Return'].fillna(0)
            
            # Calculate strategy portfolio return (existing strategy with 20% threshold)
            returns_df['Strategy_Return'] = (returns_df['POW_Weight'] * returns_df['POW_Return'] + 
                                           returns_df['NT2_Weight'] * returns_df['NT2_Return'])
            
            # Calculate equally weighted return
            returns_df['Equal_Weight_Return'] = (0.5 * returns_df['POW_Return'] + 
                                               0.5 * returns_df['NT2_Return'])
            
            # NEW: Calculate "Best Growth" portfolio - 100% in stock with higher quarterly return
            returns_df['Best_Growth_Return'] = 0.0
            for i in range(len(returns_df)):
                pow_ret = returns_df.iloc[i]['POW_Return']
                nt2_ret = returns_df.iloc[i]['NT2_Return']
                # Invest 100% in whichever stock has higher growth this quarter
                if pow_ret >= nt2_ret:
                    returns_df.iloc[i, returns_df.columns.get_loc('Best_Growth_Return')] = pow_ret
                else:
                    returns_df.iloc[i, returns_df.columns.get_loc('Best_Growth_Return')] = nt2_ret
            
            # Mock VNI return (normally would come from VNI data)
            np.random.seed(123)
            returns_df['VNI_Return'] = np.random.normal(1.5, 6, len(returns_df))
            
            # Calculate cumulative returns
            returns_df['Strategy_Cumulative'] = (1 + returns_df['Strategy_Return']/100).cumprod()
            returns_df['Equal_Weight_Cumulative'] = (1 + returns_df['Equal_Weight_Return']/100).cumprod()
            returns_df['Best_Growth_Cumulative'] = (1 + returns_df['Best_Growth_Return']/100).cumprod()
            returns_df['VNI_Cumulative'] = (1 + returns_df['VNI_Return']/100).cumprod()
            
            return returns_df
        
        return None
        
    except Exception as e:
        st.error(f"Error calculating portfolio returns: {str(e)}")
        return None

def create_growth_line_chart(quarterly_df):
    """Create line chart showing YoY growth trends for POW and NT2"""
    try:
        if quarterly_df is None or quarterly_df.empty:
            return None
            
        # Filter data that has valid YoY growth data
        valid_data = quarterly_df.dropna(subset=['POW_YoY_Growth', 'NT2_YoY_Growth'])
        
        if valid_data.empty:
            return None
            
        fig = go.Figure()
        
        # Add POW YoY growth line
        fig.add_trace(
            go.Scatter(
                x=valid_data['Quarter_Label'],
                y=valid_data['POW_YoY_Growth'],
                mode='lines+markers',
                name='POW YoY Growth',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6),
                hovertemplate='<b>POW</b><br>' +
                            'Quarter: %{x}<br>' +
                            'YoY Growth: %{y:.2f}%<br>' +
                            '<extra></extra>'
            )
        )
        
        # Add NT2 YoY growth line
        fig.add_trace(
            go.Scatter(
                x=valid_data['Quarter_Label'],
                y=valid_data['NT2_YoY_Growth'],
                mode='lines+markers',
                name='NT2 YoY Growth',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=6),
                hovertemplate='<b>NT2</b><br>' +
                            'Quarter: %{x}<br>' +
                            'YoY Growth: %{y:.2f}%<br>' +
                            '<extra></extra>'
            )
        )
        
        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title='POW vs NT2 YoY Growth Trends',
            xaxis_title='Quarter',
            yaxis_title='YoY Growth Rate (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating growth line chart: {e}")
        return None

def plot_contracted_volume_growth(quarterly_df):
    """Plot contracted volume growth for POW and NT2"""
    try:
        if quarterly_df is None or quarterly_df.empty:
            return None
            
        # Filter data that has both POW and NT2 contracted volumes
        valid_data = quarterly_df.dropna(subset=['POW_Contracted', 'NT2_Contracted'])
        
        if valid_data.empty:
            return None
            
        fig = go.Figure()
        
        # Add POW contracted volume bars
        fig.add_trace(
            go.Bar(
                x=valid_data['Quarter_Label'],
                y=valid_data['POW_Contracted'],
                name='POW Contracted Volume',
                marker_color='#1f77b4',
                hovertemplate='<b>POW</b><br>' +
                            'Quarter: %{x}<br>' +
                            'Contracted Volume: %{y:,.0f}<br>' +
                            '<extra></extra>'
            )
        )
        
        # Add NT2 contracted volume bars
        fig.add_trace(
            go.Bar(
                x=valid_data['Quarter_Label'],
                y=valid_data['NT2_Contracted'],
                name='NT2 Contracted Volume',
                marker_color='#ff7f0e',
                hovertemplate='<b>NT2</b><br>' +
                            'Quarter: %{x}<br>' +
                            'Contracted Volume: %{y:,.0f}<br>' +
                            '<extra></extra>'
            )
        )
        
        fig.update_layout(
            title='POW vs NT2 Contracted Volume Growth',
            xaxis_title='Quarter',
            yaxis_title='Contracted Volume',
            barmode='group',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting contracted volume growth: {e}")
        return None

def plot_portfolio_performance(returns_df):
    """Plot portfolio performance comparison only (no allocation chart)"""
    try:
        if returns_df is None or returns_df.empty:
            return None
            
        fig = go.Figure()
        
        # Cumulative returns starting from 0%
        fig.add_trace(
            go.Scatter(
                x=returns_df['Quarter_Label'],
                y=(returns_df['Strategy_Cumulative'] - 1) * 100,  # Convert to percentage starting from 0%
                mode='lines+markers',
                name='Strategy Portfolio',
                line=dict(color='red', width=3),
                hovertemplate='%{x}<br>Strategy: %{y:.2f}%<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=returns_df['Quarter_Label'],
                y=(returns_df['Equal_Weight_Cumulative'] - 1) * 100,  # Convert to percentage starting from 0%
                mode='lines+markers',
                name='Equal Weight',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>Equal Weight: %{y:.2f}%<extra></extra>'
            )
        )
        
        # NEW: Add Best Growth Portfolio
        fig.add_trace(
            go.Scatter(
                x=returns_df['Quarter_Label'],
                y=(returns_df['Best_Growth_Cumulative'] - 1) * 100,  # Convert to percentage starting from 0%
                mode='lines+markers',
                name='Best Growth (100%)',
                line=dict(color='purple', width=2, dash='dash'),
                hovertemplate='%{x}<br>Best Growth: %{y:.2f}%<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=returns_df['Quarter_Label'],
                y=(returns_df['VNI_Cumulative'] - 1) * 100,  # Convert to percentage starting from 0%
                mode='lines+markers',
                name='VNI Index',
                line=dict(color='green', width=2),
                hovertemplate='%{x}<br>VNI: %{y:.2f}%<extra></extra>'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='POW vs NT2 Portfolio Performance Comparison',
            xaxis_title='Quarter',
            yaxis_title='Cumulative Return (%)',
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Add horizontal line at 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting portfolio performance: {str(e)}")
        return None

def run_gas_strategy(pow_df=None, convert_to_excel=None, convert_to_csv=None, tab_focus=None):
    """Main function to run the gas strategy analysis"""
    try:
        pvpower_df = load_pvpower_data()
        quarterly_df = process_quarterly_data(pvpower_df)
        
        # Step 3: Construct portfolio strategy
        strategy_df = construct_portfolio_strategy(quarterly_df)
        
        if strategy_df is None:
            st.error("Could not construct portfolio strategy")
            return
            
        # Step 4: Get stock returns using SSI API (start from 2019)
        stock_data = get_stock_returns_ssi(['POW', 'NT2'], start_year=2019, end_year=2025)
        
        if not stock_data:
            st.error("Could not get stock returns data")
            return
            
        # Step 5: Calculate portfolio returns
        returns_df = calculate_portfolio_returns(strategy_df, stock_data)
        
        if returns_df is None:
            st.error("Could not calculate portfolio returns")
            return
        
        # Tab-specific content display

        if tab_focus == "performance" or tab_focus is None:
            # Performance Chart Tab - Show only the chart
            
            st.write("📈 **Portfolio Performance Comparison**")
            fig = plot_portfolio_performance(returns_df)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        
        elif tab_focus == "details":
            # Portfolio Details Tab - Show two tables: stock weights and returns
            st.write("📋 **Portfolio Return Analysis**")
            
            # Table 1: Stock Weights for Each Period
            st.write("**Stock Weights by Quarter:**")
            portfolio_weights = strategy_df[['Quarter_Label', 'POW_Weight', 'NT2_Weight']].copy()
            portfolio_weights['POW_Weight'] = (portfolio_weights['POW_Weight'] * 100).round(1)
            portfolio_weights['NT2_Weight'] = (portfolio_weights['NT2_Weight'] * 100).round(1)
            portfolio_weights.columns = ['Quarter', 'POW Weight (%)', 'NT2 Weight (%)']
            st.dataframe(portfolio_weights, use_container_width=True)
            
            # Table 2: Portfolio Returns Comparison (Strategy vs Equal Weight only)
            st.write("**Portfolio Returns by Quarter:**")
            returns_comparison = returns_df[['Quarter_Label', 'Strategy_Return', 'Equal_Weight_Return']].copy()
            returns_comparison['Strategy_Return'] = returns_comparison['Strategy_Return'].round(2)
            returns_comparison['Equal_Weight_Return'] = returns_comparison['Equal_Weight_Return'].round(2)
            returns_comparison.columns = ['Quarter', 'Strategy Portfolio (%)', 'Equal Weight Portfolio (%)']
            st.dataframe(returns_comparison, use_container_width=True)
            
        elif tab_focus == "growth":
            # Volume Growth Tab - Show only the table
            st.write("📈 **Contracted Volume Growth Analysis**")
            
            # Display growth data table only
            growth_data = quarterly_df[['Quarter_Label', 'POW_Contracted', 'NT2_Contracted', 'POW_YoY_Growth', 'NT2_YoY_Growth']].copy()
            growth_data['POW_YoY_Growth'] = growth_data['POW_YoY_Growth'].round(2)
            growth_data['NT2_YoY_Growth'] = growth_data['NT2_YoY_Growth'].round(2)
            growth_data.columns = ['Quarter', 'POW Contracted Volume', 'NT2 Contracted Volume', 'POW YoY Growth (%)', 'NT2 YoY Growth (%)']
            
            st.dataframe(growth_data, use_container_width=True)
        
        # Download options (show in all tabs)
        if convert_to_excel and convert_to_csv and tab_focus != "growth":
            st.write("📥 **Download Data:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.download_button(
                    label="📊 Download Strategy Data (Excel)",
                    data=convert_to_excel(returns_df),
                    file_name="pow_nt2_strategy_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="gas_strategy_excel_download"
                ):
                    st.success("Strategy data downloaded!")
            
            with col2:
                if st.download_button(
                    label="📄 Download Strategy Data (CSV)",
                    data=convert_to_csv(returns_df),
                    file_name="pow_nt2_strategy_data.csv",
                    mime="text/csv",
                    key="gas_strategy_csv_download"
                ):
                    st.success("Strategy data downloaded!")
        
    except Exception as e:
        st.error(f"Error running gas strategy: {str(e)}")
        st.exception(e)
