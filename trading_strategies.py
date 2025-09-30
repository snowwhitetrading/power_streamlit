"""
Clean Trading Strategies Module
Comprehensive implementation of all power sector trading strategies
"""

import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_vni_data():
    """Load VNI data from vn_index_monthly.csv file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vni_file = os.path.join(script_dir, 'data', 'vn_index_monthly.csv')
        
        if os.path.exists(vni_file):
            df = pd.read_csv(vni_file)
            
            # Clean column names - the CSV has 'date' and 'VNINDEX' columns
            if len(df.columns) >= 2:
                df.columns = ['date', 'VNINDEX']  # Standardize column names
            
            # Remove any invalid rows
            df = df.dropna(subset=['date', 'VNINDEX'])
            df = df[~df['date'].astype(str).str.lower().isin(['date', 'period', 'time'])]
            
            # Clean VNINDEX values (remove commas and convert to float)
            df['VNINDEX'] = df['VNINDEX'].astype(str).str.replace(',', '')
            df['VNINDEX'] = pd.to_numeric(df['VNINDEX'], errors='coerce')
            df = df.dropna(subset=['VNINDEX'])
            
            # Convert date column - handle formats like "1Q2011", "2Q2011", etc.
            def convert_quarter_to_date(quarter_str):
                try:
                    if 'Q' in str(quarter_str):
                        quarter = int(quarter_str[0])
                        year = int(quarter_str[2:])
                        # Convert to end of quarter date for proper alignment
                        month = quarter * 3  # End of quarter month (3, 6, 9, 12)
                        return pd.to_datetime(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
                    else:
                        return pd.to_datetime(quarter_str)
                except:
                    return pd.NaT
            
            df['date'] = df['date'].apply(convert_quarter_to_date)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date')  # Sort by date
            df.set_index('date', inplace=True)
            
            # Calculate quarterly returns
            df['Quarter_Return'] = df['VNINDEX'].pct_change() * 100
            df['Cumulative_Return'] = (1 + df['Quarter_Return']/100).cumprod() * 100 - 100
            
            # Fill first quarter return with 0
            df['Quarter_Return'].fillna(0, inplace=True)
            df['Cumulative_Return'].fillna(0, inplace=True)
            
            st.info(f"✅ Loaded VNI data: {len(df)} quarters from {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
            
            return df
        else:
            st.error("VNI data file (vn_index_monthly.csv) not found")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading VNI data: {e}")
        return pd.DataFrame()

def load_enso_data():
    """Load ENSO/ONI data from enso_data_quarterly.csv file"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        enso_file = os.path.join(script_dir, 'data', 'enso_data_quarterly.csv')
        
        if os.path.exists(enso_file):
            df = pd.read_csv(enso_file)
            
            # Convert date column - handle formats like "1Q11", "2Q11", etc.
            def convert_quarter_to_date(quarter_str):
                try:
                    if 'Q' in str(quarter_str):
                        quarter = int(quarter_str[0])
                        year_str = quarter_str[2:]
                        year = int(f"20{year_str}") if len(year_str) == 2 else int(year_str)
                        month = quarter * 3 - 2  # Q1=1, Q2=4, Q3=7, Q4=10
                        return pd.to_datetime(f"{year}-{month:02d}-01")
                    else:
                        return pd.to_datetime(quarter_str)
                except:
                    return pd.NaT
            
            df['date'] = df['date'].apply(convert_quarter_to_date)
            df = df.dropna(subset=['date'])
            df.set_index('date', inplace=True)
            
            return df
        else:
            st.error("ENSO data file (enso_data_quarterly.csv) not found")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading ENSO data: {e}")
        return pd.DataFrame()
        
        # Convert date column - handle quarterly format like "1Q2011"
        if vni_df[date_col].dtype == 'object' and 'Q' in str(vni_df[date_col].iloc[0]):
            # Convert quarterly format to date
            def quarter_to_date(quarter_str):
                try:
                    q, year = quarter_str.split('Q')
                    month = int(q) * 3 - 2  # 1Q->1, 2Q->4, 3Q->7, 4Q->10
                    return pd.to_datetime(f"{year}-{month:02d}-01")
                except:
                    return pd.to_datetime(quarter_str)
            
            vni_df['Date'] = vni_df[date_col].apply(quarter_to_date)
        else:
            vni_df['Date'] = pd.to_datetime(vni_df[date_col])
        
        # Filter from 1Q2011
        start_date = pd.to_datetime('2011-01-01')
        vni_df = vni_df[vni_df['Date'] >= start_date].copy()
        
        # Calculate quarterly returns
        vni_df = vni_df.sort_values('Date')
        vni_df['Quarter_Return'] = vni_df[value_col].pct_change() * 100
        vni_df['Quarter_Return'] = vni_df['Quarter_Return'].fillna(0)
        vni_df['Cumulative_Return'] = (1 + vni_df['Quarter_Return']/100).cumprod() * 100 - 100
        
        # Set Date as index for easier manipulation
        vni_df.set_index('Date', inplace=True)
        
        return vni_df
        
    except Exception as e:
        error_msg = f"Error loading VNI data: {e}"
        print(error_msg)
        try:
            st.error(error_msg)
        except:
            pass
        # Return empty DataFrame instead of mock data - VNI should use real data only
        return pd.DataFrame()

def get_all_power_stocks():
    """Get all power stocks from hydro, gas, and coal strategies"""
    all_stocks = set()
    
    # Get hydro stocks from water_list.csv
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        water_list_file = os.path.join(script_dir, 'data', 'water_list.csv')
        if os.path.exists(water_list_file):
            lake_df = pd.read_csv(water_list_file)
            for col in lake_df.columns:
                stocks = lake_df[col].dropna().astype(str)
                for stock in stocks:
                    stock = stock.strip()
                    if len(stock) == 3 and stock.isalpha() and stock.isupper():
                        all_stocks.add(stock)
    except Exception as e:
        st.warning(f"Could not load hydro stocks: {e}")
    
    # Add comprehensive hydro stocks list
    hydro_stocks = ['REE', 'PC1', 'HDG', 'GEG', 'TTA', 'AVC', 'GHC', 'VPD', 'DRL', 'S4A', 'SBA', 'VSH', 'NED', 'TMP', 'HNA', 'SHP']
    all_stocks.update(hydro_stocks)
    
    # Add known gas stocks
    gas_stocks = ['POW', 'NT2']
    all_stocks.update(gas_stocks)
    
    # Add known coal stocks  
    coal_stocks = ['QTP', 'PPC', 'HND']
    all_stocks.update(coal_stocks)
    
    st.info(f"Found {len(all_stocks)} power stocks: {sorted(list(all_stocks))}")
    return sorted(list(all_stocks))

def get_equal_weighted_portfolio_return():
    """Get equally weighted portfolio return for all power stocks using SSI API"""
    try:
        # Get all power stocks
        all_stocks = get_all_power_stocks()
        
        if not all_stocks:
            st.warning("No power stocks found, using VNI as proxy")
            return load_vni_data()
        
        # Try to get stock data using SSI API
        try:
            # Try importing ssi_api with different methods
            ssi_module = None
            try:
                from . import ssi_api
                ssi_module = ssi_api
            except ImportError:
                try:
                    import ssi_api
                    ssi_module = ssi_api
                except ImportError:
                    # Try importing as module in current directory
                    import sys
                    import os
                    sys.path.append(os.path.dirname(__file__))
                    try:
                        import ssi_api
                        ssi_module = ssi_api
                    except ImportError:
                        ssi_module = None
            
            if ssi_module and hasattr(ssi_module, 'get_quarterly_stock_data'):
                # Get quarterly stock data from 2011 to 2025
                stock_data = ssi_module.get_quarterly_stock_data(all_stocks, '2011-01-01', '2025-06-30')
            
            if stock_data and isinstance(stock_data, dict) and len(stock_data) > 0:
                # Create a combined DataFrame with quarterly returns for each stock
                quarterly_returns = {}
                
                for symbol in stock_data.keys():
                    df = stock_data[symbol]
                    if df is not None and not df.empty and 'close' in df.columns:
                        try:
                            # Make a copy to avoid modifying original data
                            df = df.copy()
                            
                            # Ensure date column is datetime
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.sort_values('date')
                            
                            # Calculate quarterly returns
                            df['quarterly_return'] = df['close'].pct_change() * 100
                            df['quarterly_return'] = df['quarterly_return'].fillna(0)
                            
                            # Create period labels safely
                            def create_period_label(date_val):
                                try:
                                    if hasattr(date_val, 'year') and hasattr(date_val, 'month'):
                                        return f"{date_val.year}Q{(date_val.month-1)//3 + 1}"
                                    else:
                                        return None
                                except:
                                    return None
                            
                            df['period'] = df['date'].apply(create_period_label)
                            
                            # Store quarterly returns
                            for _, row in df.iterrows():
                                period = row['period']
                                # Ensure period is a string, not a list or other type
                                if isinstance(period, (list, tuple)):
                                    period = str(period[0]) if period else None
                                elif period is not None:
                                    period = str(period)
                                    
                                if period and period not in quarterly_returns:
                                    quarterly_returns[period] = {}
                                if period:
                                    quarterly_returns[period][symbol] = float(row['quarterly_return'])
                        except Exception as e:
                            st.warning(f"Error processing {symbol}: {e}")
                            continue
                
                if quarterly_returns:
                    # Create equally weighted portfolio returns
                    result_data = []
                    for period in sorted(quarterly_returns.keys()):
                        period_data = quarterly_returns[period]
                        if period_data:
                            # Calculate equal weight return (average of all available stocks)
                            equal_return = sum(period_data.values()) / len(period_data)
                            result_data.append({
                                'period': period,
                                'Quarter_Return': equal_return
                            })
                    
                    if result_data:
                        result_df = pd.DataFrame(result_data)
                        
                        # Convert period to date
                        def period_to_date(period_str):
                            try:
                                year = int(period_str[:4])
                                quarter = int(period_str[5])
                                month = quarter * 3
                                return pd.to_datetime(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
                            except:
                                return pd.to_datetime('2011-01-01')
                        
                        result_df['date'] = result_df['period'].apply(period_to_date)
                        result_df = result_df.set_index('date')
                        result_df = result_df.sort_index()
                        
                        # Calculate cumulative returns
                        result_df['Cumulative_Return'] = (1 + result_df['Quarter_Return']/100).cumprod() * 100 - 100
                        
                        st.success(f"✅ Loaded equal weighted portfolio with {len(all_stocks)} stocks, {len(result_df)} quarters")
                        return result_df
                        
        except ImportError:
            st.warning("SSI API not available, using VNI as proxy")
        except Exception as e:
            st.warning(f"Error getting stock data from SSI: {e}")
        
        # Fallback: Use VNI data as proxy for equal weighted portfolio
        vni_df = load_vni_data()
        if not vni_df.empty:
            st.info("Using VNI data as proxy for equally weighted power portfolio")
            return vni_df.copy()
        
        # Last resort: return empty DataFrame
        st.error("No data available for equally weighted portfolio")
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error creating equally weighted portfolio: {e}")
        return pd.DataFrame()
        
        # Fallback: create conservative baseline using VNI data
        try:
            vni_data = load_vni_data()
            if not vni_data.empty and 'Quarter_Return' in vni_data.columns:
                # Use VNI returns as baseline but make more conservative
                date_range = pd.date_range('2011-01-01', '2025-09-30', freq='QE')
                result_df = pd.DataFrame(index=date_range)
                
                # Map VNI returns and make them more conservative for equal weight
                vni_aligned = vni_data.reindex(date_range)['Quarter_Return'].fillna(0) * 0.8
                result_df['Quarter_Return'] = vni_aligned
                result_df['Cumulative_Return'] = (1 + result_df['Quarter_Return']/100).cumprod() * 100 - 100
                return result_df
        except:
            pass
        
        # If no real data available, return empty DataFrame
        st.error("No real equal weighted portfolio data found")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error calculating equal-weighted portfolio: {e}")
        return pd.DataFrame()

def get_hydro_flood_portfolio_return():
    """Get hydro flood level portfolio return using hydro_strategy module - flood level cumulative returns"""
    try:
        # Import hydro_strategy module to get flood level portfolio directly
        from hydro_strategy import (
            load_water_reservoir_data, 
            load_stock_mappings, 
            get_quarterly_returns,
            calculate_quarterly_growth_data,
            create_portfolios
        )
        
        # Load hydro data and create portfolios
        reservoir_df = load_water_reservoir_data()
        mappings_result = load_stock_mappings()
        
        if isinstance(mappings_result, tuple):
            mappings, liquid_stocks, illiquid_stocks = mappings_result
        else:
            mappings = mappings_result
        
        quarterly_returns = get_quarterly_returns(mappings)
        growth_data = calculate_quarterly_growth_data(reservoir_df, mappings)
        portfolios = create_portfolios(growth_data, quarterly_returns)
        
        # Get the flood level portfolio specifically - use cumulative returns directly
        if isinstance(portfolios, dict) and 'flood_level' in portfolios:
            flood_level_df = portfolios['flood_level']
            
            # Use the cumulative returns directly from the flood level portfolio
            if hasattr(flood_level_df, 'empty') and not flood_level_df.empty:
                # Return the flood level portfolio with cumulative returns
                result_df = flood_level_df.copy()
                
                # Ensure we have the right column names for trading strategies
                if 'cumulative_return' in result_df.columns:
                    result_df['Cumulative_Return'] = result_df['cumulative_return']
                if 'quarterly_return' in result_df.columns:
                    result_df['Quarter_Return'] = result_df['quarterly_return']
                    
                # Convert period to datetime index if needed
                if 'period' in result_df.columns:
                    def period_to_date(period_str):
                        try:
                            if 'Q' in str(period_str):
                                year = int(period_str[:4])
                                quarter = int(period_str[5])
                                month = quarter * 3  # End of quarter month
                                return pd.to_datetime(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
                            return pd.to_datetime(period_str)
                        except:
                            return pd.to_datetime('2011-01-01')
                    
                    result_df['Date'] = result_df['period'].apply(period_to_date)
                    result_df = result_df.set_index('Date')
                
                st.success(f"✅ Loaded hydro flood level portfolio with {len(result_df)} periods")
                return result_df
        
        # If no flood level portfolio data, return empty DataFrame
        st.warning("No hydro flood level portfolio data available from strategy module")
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading hydro flood portfolio: {e}")
        return pd.DataFrame()

def get_gas_contracted_volume_return():
    """Get gas contracted volume portfolio return using gas_strategy module - concentrated portfolio (Best Growth) returns"""
    try:
        # Import gas_strategy module to get concentrated portfolio (Best Growth) directly
        from gas_strategy import (
            load_pvpower_data,
            process_quarterly_data,
            construct_portfolio_strategy,
            get_stock_returns_ssi,
            calculate_portfolio_returns
        )
        
        # Run the gas strategy to get portfolio returns
        pvpower_df = load_pvpower_data()
        if pvpower_df is None or pvpower_df.empty:
            st.warning("No PVPower data available for gas strategy")
            return pd.DataFrame()
        
        # Process quarterly data
        quarterly_df = process_quarterly_data(pvpower_df)
        if quarterly_df is None or quarterly_df.empty:
            st.warning("No quarterly data available for gas strategy")
            return pd.DataFrame()
        
        # Construct portfolio strategy
        strategy_df = construct_portfolio_strategy(quarterly_df)
        if strategy_df is None or strategy_df.empty:
            st.warning("No strategy data available for gas strategy")
            return pd.DataFrame()
        
        # Get stock returns
        gas_stocks = ['POW', 'NT2']
        try:
            stock_data = get_stock_returns_ssi(gas_stocks, start_year=2019, end_year=2025)
        except Exception as e:
            st.warning(f"Error getting stock data from gas strategy: {e}")
            return pd.DataFrame()
        
        if not stock_data:
            st.warning("No stock data available for gas strategy")
            return pd.DataFrame()
        
        # Calculate portfolio returns (this includes Best_Growth_Return which is our concentrated portfolio)
        returns_df = calculate_portfolio_returns(strategy_df, stock_data)
        
        if returns_df is not None and not returns_df.empty and 'Best_Growth_Return' in returns_df.columns:
            # Use the Best Growth (concentrated) portfolio returns directly
            result_df = returns_df[['Quarter_Label', 'Best_Growth_Return', 'Best_Growth_Cumulative']].copy()
            result_df = result_df.rename(columns={
                'Quarter_Label': 'period',
                'Best_Growth_Return': 'quarterly_return', 
                'Best_Growth_Cumulative': 'cumulative_return'
            })
            
            # Ensure we have the right column names for trading strategies
            result_df['Quarter_Return'] = result_df['quarterly_return']
            result_df['Cumulative_Return'] = (result_df['cumulative_return'] - 1) * 100  # Convert to percentage
            
            # Convert period to datetime index if needed
            if 'period' in result_df.columns:
                def period_to_date(period_str):
                    try:
                        if 'Q' in str(period_str):
                            year = int(period_str[:4])
                            quarter = int(period_str[5])
                            month = quarter * 3  # End of quarter month
                            return pd.to_datetime(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
                        return pd.to_datetime(period_str)
                    except:
                        return pd.to_datetime('2019-01-01')
                
                result_df['Date'] = result_df['period'].apply(period_to_date)
                result_df = result_df.set_index('Date')
            
            st.success(f"✅ Loaded gas concentrated portfolio (Best Growth) with {len(result_df)} periods")
            return result_df
        
        # If no concentrated portfolio data, return empty DataFrame
        st.warning("No gas concentrated portfolio data available from strategy module")
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading gas concentrated portfolio: {e}")
        return pd.DataFrame()

def get_coal_volume_growth_return():
    """Get coal volume growth return using coal_strategy module - concentrated portfolio returns"""
    try:
        # Import coal_strategy module to get concentrated portfolio directly
        from coal_strategy import (
            load_coal_volume_data,
            calculate_yoy_growth,
            fetch_stock_data,
            convert_to_quarterly_returns,
            create_coal_portfolios,
            calculate_cumulative_returns
        )
        
        # Run the coal strategy to get portfolios
        coal_df = load_coal_volume_data()
        if coal_df.empty:
            st.warning("No coal volume data available")
            return pd.DataFrame()
        
        # Calculate growth and get stock data
        growth_data = calculate_yoy_growth(coal_df)
        coal_stocks = ['PPC', 'QTP', 'HND']
        stock_data = fetch_stock_data(coal_stocks)
        
        if not stock_data:
            st.warning("No stock data available for coal strategy")
            return pd.DataFrame()
        
        # Convert to quarterly returns and create portfolios
        quarterly_returns = convert_to_quarterly_returns(stock_data)
        portfolios = create_coal_portfolios(growth_data, quarterly_returns)
        
        # Get the CONCENTRATED portfolio specifically (portfolios['concentrated'])
        if portfolios and 'concentrated' in portfolios:
            concentrated_portfolio_data = portfolios['concentrated']
            
            # Calculate cumulative returns for concentrated portfolio
            concentrated_df = calculate_cumulative_returns(concentrated_portfolio_data, start_period='2018Q4')
            
            if not concentrated_df.empty and hasattr(concentrated_df, 'columns'):
                # Use the concentrated portfolio cumulative returns directly
                result_df = concentrated_df.copy()
                
                # Ensure we have the right column names for trading strategies
                if 'cumulative_return' in result_df.columns:
                    result_df['Cumulative_Return'] = result_df['cumulative_return']
                if 'quarterly_return' in result_df.columns:
                    result_df['Quarter_Return'] = result_df['quarterly_return']
                    
                # Convert period to datetime index if needed
                if 'period' in result_df.columns:
                    def period_to_date(period_str):
                        try:
                            if 'Q' in str(period_str):
                                year = int(period_str[:4])
                                quarter = int(period_str[5])
                                month = quarter * 3  # End of quarter month
                                return pd.to_datetime(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
                            return pd.to_datetime(period_str)
                        except:
                            return pd.to_datetime('2019-01-01')
                    
                    result_df['Date'] = result_df['period'].apply(period_to_date)
                    result_df = result_df.set_index('Date')
                
                st.success(f"✅ Loaded coal concentrated portfolio with {len(result_df)} periods")
                return result_df
        
        # If no concentrated portfolio data, return empty DataFrame
        st.warning("No coal concentrated portfolio data available from strategy module")
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading coal concentrated portfolio: {e}")
        return pd.DataFrame()

def calculate_oni_based_strategy(enso_df=None):
    """Calculate ONI-based strategy using equal weighted portfolios for each sector"""
    try:
        # Load ENSO data if not provided
        if enso_df is None or enso_df.empty:
            enso_df = load_enso_data()
        
        if enso_df.empty:
            st.error("No ENSO data available for ONI strategy")
            return pd.DataFrame()
        
        # Get equal weighted portfolio for each sector
        st.info("Loading equal weighted portfolio data for ONI strategy...")
        
        # Define sector stocks for equal weighted portfolios
        hydro_stocks = ['REE', 'PC1', 'HDG', 'GEG', 'TTA', 'AVC', 'GHC', 'VPD', 'DRL', 'S4A', 'SBA', 'VSH', 'NED', 'TMP', 'HNA', 'SHP']
        gas_stocks = ['POW', 'NT2']
        coal_stocks = ['QTP', 'PPC', 'HND']
        
        # Create date range
        date_range = pd.date_range('2011-01-01', '2025-06-30', freq='QE')
        result_df = pd.DataFrame(index=date_range)
        
        # Get equal weighted returns for each sector using SSI API
        try:
            # Try importing ssi_api with different methods
            ssi_module = None
            try:
                from . import ssi_api
                ssi_module = ssi_api
            except ImportError:
                try:
                    import ssi_api
                    ssi_module = ssi_api
                except ImportError:
                    import sys
                    import os
                    sys.path.append(os.path.dirname(__file__))
                    try:
                        import ssi_api
                        ssi_module = ssi_api
                    except ImportError:
                        ssi_module = None
            
            sector_returns = {}
            
            if ssi_module and hasattr(ssi_module, 'get_quarterly_stock_data'):
                # Get quarterly returns for each sector
                for sector_name, stocks in [('hydro', hydro_stocks), ('gas', gas_stocks), ('coal', coal_stocks)]:
                    try:
                        stock_data = ssi_module.get_quarterly_stock_data(stocks, '2011-01-01', '2025-06-30')
                        
                        if stock_data and isinstance(stock_data, dict) and len(stock_data) > 0:
                            quarterly_returns = {}
                            
                            for symbol in stock_data.keys():
                                df = stock_data[symbol]
                                if df is not None and not df.empty and 'close' in df.columns:
                                    try:
                                        df = df.copy()
                                        df['date'] = pd.to_datetime(df['date'])
                                        df = df.sort_values('date')
                                        df['quarterly_return'] = df['close'].pct_change() * 100
                                        df['quarterly_return'] = df['quarterly_return'].fillna(0)
                                        
                                        def create_period_label(date_val):
                                            try:
                                                return f"{date_val.year}Q{(date_val.month-1)//3 + 1}"
                                            except:
                                                return None
                                        
                                        df['period'] = df['date'].apply(create_period_label)
                                        
                                        for _, row in df.iterrows():
                                            period = row['period']
                                            if period and period not in quarterly_returns:
                                                quarterly_returns[period] = {}
                                            if period:
                                                quarterly_returns[period][symbol] = float(row['quarterly_return'])
                                    except Exception as e:
                                        continue
                            
                            # Calculate equal weighted returns for this sector
                            sector_data = []
                            for period in sorted(quarterly_returns.keys()):
                                period_data = quarterly_returns[period]
                                if period_data:
                                    equal_return = sum(period_data.values()) / len(period_data)
                                    sector_data.append({
                                        'period': period,
                                        'Quarter_Return': equal_return
                                    })
                            
                            if sector_data:
                                sector_df = pd.DataFrame(sector_data)
                                def period_to_date(period_str):
                                    try:
                                        year = int(period_str[:4])
                                        quarter = int(period_str[5])
                                        month = quarter * 3
                                        return pd.to_datetime(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
                                    except:
                                        return pd.to_datetime('2011-01-01')
                                
                                sector_df['date'] = sector_df['period'].apply(period_to_date)
                                sector_df = sector_df.set_index('date')
                                sector_returns[sector_name] = sector_df
                                
                    except Exception as e:
                        st.warning(f"Error getting {sector_name} sector data: {e}")
                        continue
            else:
                st.warning("SSI API not available for ONI strategy")
        
        except Exception as e:
            st.warning(f"Error loading sector data: {e}")
        
        # Calculate ONI strategy returns
        returns = []
        for date in date_range:
            # Get ONI value for this quarter
            oni_val = 0
            year = date.year
            quarter = (date.month - 1) // 3 + 1
            
            # Find matching ONI value
            oni_found = False
            for enso_date, row in enso_df.iterrows():
                enso_year = enso_date.year
                enso_quarter = (enso_date.month - 1) // 3 + 1
                
                if enso_year == year and enso_quarter == quarter:
                    oni_val = row['ONI'] if 'ONI' in row else 0
                    oni_found = True
                    break
            
            if not oni_found:
                closest_date = min(enso_df.index, key=lambda x: abs((x - date).days))
                if abs((closest_date - date).days) < 100:
                    oni_val = enso_df.loc[closest_date, 'ONI'] if 'ONI' in enso_df.columns else 0
            
            # Get sector returns for this date
            hydro_ret = 0
            gas_ret = 0
            coal_ret = 0
            
            for sector_name, sector_data in sector_returns.items():
                if not sector_data.empty:
                    if date in sector_data.index:
                        ret = sector_data.loc[date, 'Quarter_Return']
                    else:
                        # Find closest date
                        closest_sector_date = min(sector_data.index, key=lambda x: abs((x - date).days))
                        if abs((closest_sector_date - date).days) < 100:
                            ret = sector_data.loc[closest_sector_date, 'Quarter_Return']
                        else:
                            ret = 0
                    
                    if sector_name == 'hydro':
                        hydro_ret = ret
                    elif sector_name == 'gas':
                        gas_ret = ret
                    elif sector_name == 'coal':
                        coal_ret = ret
            
            # Apply ONI-based allocation strategy
            if oni_val > 0.5:
                # ONI > 0.5: invest 50%/50% in coal/gas equally
                weighted_return = 0.5 * gas_ret + 0.5 * coal_ret
            elif oni_val < -0.5:
                # ONI < -0.5: invest 100% in hydro equal weighted portfolio
                weighted_return = hydro_ret
            else:
                # -0.5 <= ONI <= 0.5: invest 50%/25%/25% in hydro/coal/gas
                weighted_return = 0.5 * hydro_ret + 0.25 * gas_ret + 0.25 * coal_ret
            
            returns.append(weighted_return)
        
        result_df['Quarter_Return'] = returns
        result_df['Cumulative_Return'] = (1 + result_df['Quarter_Return']/100).cumprod() * 100 - 100
        
        st.success(f"✅ Calculated ONI strategy with {len(result_df)} quarters")
        return result_df
        
    except Exception as e:
        st.error(f"Error calculating ONI-based strategy: {e}")
        return pd.DataFrame()

def calculate_alpha_strategy(enso_df=None):
    """Calculate Alpha strategy with timeline implementation and ONI conditions using real data"""
    try:
        # Load ENSO data for ONI values if not provided
        if enso_df is None or enso_df.empty:
            enso_df = load_enso_data()
        
        if enso_df.empty:
            st.error("No ENSO data available for Alpha strategy")
            return pd.DataFrame()
        
        # Get portfolio components
        st.info("Loading portfolio data for Alpha strategy...")
        equal_weighted = get_equal_weighted_portfolio_return()
        hydro_data = get_hydro_flood_portfolio_return()
        gas_data = get_gas_contracted_volume_return()
        coal_data = get_coal_volume_growth_return()
        
        # Also get ONI strategy for before 1Q2019 (Alpha = ONI rule)
        oni_strategy = calculate_oni_based_strategy(enso_df)
        
        # Create quarterly date range from 1Q2011 to 2Q2025
        date_range = pd.date_range('2011-01-01', '2025-06-30', freq='QE')
        result_df = pd.DataFrame(index=date_range)
        
        returns = []
        for i, date in enumerate(date_range):
            # Before 1Q2019: Alpha should equal ONI
            if date < pd.to_datetime('2019-01-01'):
                # Use ONI strategy return for this period
                if not oni_strategy.empty and date in oni_strategy.index:
                    quarterly_return = oni_strategy.loc[date, 'Quarter_Return']
                else:
                    quarterly_return = 0
                returns.append(quarterly_return)
                continue
            
            # From 1Q2019 onwards: Use complex Alpha strategy
            # Get ONI value for this quarter
            oni_val = 0
            year = date.year
            quarter = (date.month - 1) // 3 + 1
            
            # Find matching ONI value
            oni_found = False
            for enso_date, row in enso_df.iterrows():
                enso_year = enso_date.year
                enso_quarter = (enso_date.month - 1) // 3 + 1
                
                if enso_year == year and enso_quarter == quarter:
                    oni_val = row['ONI'] if 'ONI' in row else 0
                    oni_found = True
                    break
            
            if not oni_found:
                closest_date = min(enso_df.index, key=lambda x: abs((x - date).days))
                if abs((closest_date - date).days) < 100:
                    oni_val = enso_df.loc[closest_date, 'ONI'] if 'ONI' in enso_df.columns else 0
            
            # Apply Alpha strategy based on ONI conditions and timeline
            quarterly_return = 0
            
            if oni_val > 0.5:
                # ONI > 0.5: invest 50% in gas and 50% in coal
                # From 1Q2019 onwards: use specialized portfolios
                gas_ret = 0
                coal_ret = 0
                
                # Get gas return using specialized portfolio (gas_strategy.py result)
                if not gas_data.empty and 'Quarter_Return' in gas_data.columns:
                    if date in gas_data.index:
                        gas_ret = gas_data.loc[date, 'Quarter_Return']
                    else:
                        closest_gas = min(gas_data.index, key=lambda x: abs((x - date).days))
                        if abs((closest_gas - date).days) < 100:
                            gas_ret = gas_data.loc[closest_gas, 'Quarter_Return']
                
                # Get coal return using specialized portfolio (coal_strategy.py result)
                if not coal_data.empty and 'Quarter_Return' in coal_data.columns:
                    if date in coal_data.index:
                        coal_ret = coal_data.loc[date, 'Quarter_Return']
                    else:
                        closest_coal = min(coal_data.index, key=lambda x: abs((x - date).days))
                        if abs((closest_coal - date).days) < 100:
                            coal_ret = coal_data.loc[closest_coal, 'Quarter_Return']
                
                quarterly_return = 0.5 * gas_ret + 0.5 * coal_ret
                    
            elif oni_val < -0.5:
                # ONI < -0.5: invest 100% in hydro
                if date < pd.to_datetime('2020-04-01'):
                    # 1Q2011 to 1Q2020: use equally weighted portfolio
                    if not equal_weighted.empty and i < len(equal_weighted) and 'Quarter_Return' in equal_weighted.columns:
                        quarterly_return = equal_weighted.iloc[i]['Quarter_Return']
                else:
                    # 2Q2020 onwards: use hydro flood level portfolio (hydro_strategy.py result)
                    if not hydro_data.empty and 'Quarter_Return' in hydro_data.columns:
                        if date in hydro_data.index:
                            quarterly_return = hydro_data.loc[date, 'Quarter_Return']
                        else:
                            closest_hydro = min(hydro_data.index, key=lambda x: abs((x - date).days))
                            if abs((closest_hydro - date).days) < 100:
                                quarterly_return = hydro_data.loc[closest_hydro, 'Quarter_Return']
                        
            else:
                # -0.5 <= ONI <= 0.5: invest 50%/25%/25% in hydro/coal/gas
                hydro_ret = 0
                gas_ret = 0
                coal_ret = 0
                
                # Get gas return using specialized portfolio
                if not gas_data.empty and 'Quarter_Return' in gas_data.columns:
                    if date in gas_data.index:
                        gas_ret = gas_data.loc[date, 'Quarter_Return']
                    else:
                        closest_gas = min(gas_data.index, key=lambda x: abs((x - date).days))
                        if abs((closest_gas - date).days) < 100:
                            gas_ret = gas_data.loc[closest_gas, 'Quarter_Return']
                
                # Get coal return using specialized portfolio
                if not coal_data.empty and 'Quarter_Return' in coal_data.columns:
                    if date in coal_data.index:
                        coal_ret = coal_data.loc[date, 'Quarter_Return']
                    else:
                        closest_coal = min(coal_data.index, key=lambda x: abs((x - date).days))
                        if abs((closest_coal - date).days) < 100:
                            coal_ret = coal_data.loc[closest_coal, 'Quarter_Return']
                
                # For hydro: use specialized data from 2Q2020, equal weighted before
                if date >= pd.to_datetime('2020-04-01'):
                    # Use hydro flood level portfolio
                    if not hydro_data.empty and 'Quarter_Return' in hydro_data.columns:
                        if date in hydro_data.index:
                            hydro_ret = hydro_data.loc[date, 'Quarter_Return']
                        else:
                            closest_hydro = min(hydro_data.index, key=lambda x: abs((x - date).days))
                            if abs((closest_hydro - date).days) < 100:
                                hydro_ret = hydro_data.loc[closest_hydro, 'Quarter_Return']
                else:
                    # Before 2Q2020, use equal weighted for hydro portion
                    if not equal_weighted.empty and i < len(equal_weighted) and 'Quarter_Return' in equal_weighted.columns:
                        hydro_ret = equal_weighted.iloc[i]['Quarter_Return']
                
                quarterly_return = 0.5 * hydro_ret + 0.25 * coal_ret + 0.25 * gas_ret
            
            returns.append(quarterly_return)
        
        # Create result DataFrame
        result_df['Quarter_Return'] = returns
        result_df['Cumulative_Return'] = (1 + result_df['Quarter_Return']/100).cumprod() * 100 - 100
        
        st.success(f"✅ Calculated Alpha strategy with {len(result_df)} quarters (Alpha = ONI before 1Q2019)")
        return result_df
        
    except Exception as e:
        st.error(f"Error calculating Alpha strategy: {e}")
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error calculating Alpha strategy: {e}")
        return pd.DataFrame()

def create_comprehensive_strategy_comparison(enso_df=None):
    """Create comprehensive comparison of all strategies using real data"""
    try:
        # Show info message if in Streamlit context
        try:
            st.info("📊 Generating comprehensive strategy comparison...")
        except:
            print("📊 Generating comprehensive strategy comparison...")
        
        # Get all strategies
        vni_data = load_vni_data()
        equal_data = get_equal_weighted_portfolio_return()
        oni_data = calculate_oni_based_strategy()
        alpha_data = calculate_alpha_strategy()
        
        # Create unified DataFrame
        date_range = pd.date_range('2011-01-01', '2025-06-30', freq='QE')
        
        unified_df = pd.DataFrame(index=date_range)
        # Convert to string format for better chart compatibility
        unified_df['Period'] = date_range.strftime('%Y-%m-%d')
        
        # Add all strategy data with proper error handling and type conversion
        if not vni_data.empty and 'Quarter_Return' in vni_data.columns:
            try:
                vni_returns = vni_data.reindex(date_range)['Quarter_Return']
                vni_cumulative = vni_data.reindex(date_range)['Cumulative_Return'] 
                
                # Force numeric conversion with proper error handling
                unified_df['VNI_Return'] = pd.to_numeric(vni_returns.astype(str), errors='coerce').fillna(0.0)
                unified_df['VNI_Cumulative'] = pd.to_numeric(vni_cumulative.astype(str), errors='coerce').fillna(0.0)
            except Exception as e:
                st.warning(f"Error processing VNI data: {e}")
                unified_df['VNI_Return'] = 0.0
                unified_df['VNI_Cumulative'] = 0.0
        else:
            unified_df['VNI_Return'] = 0.0
            unified_df['VNI_Cumulative'] = 0.0
            
        if not equal_data.empty and 'Quarter_Return' in equal_data.columns:
            try:
                equal_returns = equal_data.reindex(date_range)['Quarter_Return']
                equal_cumulative = equal_data.reindex(date_range)['Cumulative_Return']
                
                unified_df['Equal_Return'] = pd.to_numeric(equal_returns.astype(str), errors='coerce').fillna(0.0)
                unified_df['Equal_Cumulative'] = pd.to_numeric(equal_cumulative.astype(str), errors='coerce').fillna(0.0)
            except Exception as e:
                st.warning(f"Error processing Equal Weight data: {e}")
                unified_df['Equal_Return'] = 0.0
                unified_df['Equal_Cumulative'] = 0.0
        else:
            unified_df['Equal_Return'] = 0.0
            unified_df['Equal_Cumulative'] = 0.0
            
        if not oni_data.empty and 'Quarter_Return' in oni_data.columns:
            try:
                oni_returns = oni_data.reindex(date_range)['Quarter_Return']
                oni_cumulative = oni_data.reindex(date_range)['Cumulative_Return']
                
                unified_df['ONI_Return'] = pd.to_numeric(oni_returns.astype(str), errors='coerce').fillna(0.0)
                unified_df['ONI_Cumulative'] = pd.to_numeric(oni_cumulative.astype(str), errors='coerce').fillna(0.0)
            except Exception as e:
                st.warning(f"Error processing ONI data: {e}")
                unified_df['ONI_Return'] = 0.0
                unified_df['ONI_Cumulative'] = 0.0
        else:
            unified_df['ONI_Return'] = 0.0
            unified_df['ONI_Cumulative'] = 0.0
            
        if not alpha_data.empty and 'Quarter_Return' in alpha_data.columns:
            try:
                alpha_returns = alpha_data.reindex(date_range)['Quarter_Return']
                alpha_cumulative = alpha_data.reindex(date_range)['Cumulative_Return']
                
                unified_df['Alpha_Return'] = pd.to_numeric(alpha_returns.astype(str), errors='coerce').fillna(0.0)
                unified_df['Alpha_Cumulative'] = pd.to_numeric(alpha_cumulative.astype(str), errors='coerce').fillna(0.0)
            except Exception as e:
                st.warning(f"Error processing Alpha data: {e}")
                unified_df['Alpha_Return'] = 0.0
                unified_df['Alpha_Cumulative'] = 0.0
        else:
            unified_df['Alpha_Return'] = 0.0
            unified_df['Alpha_Cumulative'] = 0.0
        
        return unified_df
        
    except Exception as e:
        try:
            st.error(f"Error creating comprehensive strategy comparison: {e}")
        except:
            print(f"Error creating comprehensive strategy comparison: {e}")
        return pd.DataFrame()

def create_unified_strategy_chart(unified_df):
    """Create unified strategy performance chart with robust data type handling"""
    try:
        if unified_df is None or unified_df.empty:
            st.warning("No data available for chart creation")
            return None
        
        # Debug: Check what columns we actually have
        st.info(f"Available columns: {list(unified_df.columns)}")
        
        # Check if required columns exist
        required_cols = ['Period', 'Alpha_Cumulative', 'ONI_Cumulative', 'Equal_Cumulative', 'VNI_Cumulative']
        missing_cols = [col for col in required_cols if col not in unified_df.columns]
        if missing_cols:
            error_msg = f"Missing columns in data: {missing_cols}"
            st.error(error_msg)
            return None
        
        # Create a clean copy of the data to avoid modifying original
        chart_df = unified_df.copy()
        
        # Robust data cleaning function
        def clean_and_convert_column(col_name):
            """Clean and convert column to numeric, handling all edge cases"""
            try:
                col_data = chart_df[col_name]
                
                # If it's already numeric, just fill NaN with 0
                if pd.api.types.is_numeric_dtype(col_data):
                    return pd.to_numeric(col_data, errors='coerce').fillna(0.0)
                
                # If it's object/string, try various conversions
                if col_data.dtype == 'object':
                    # First, convert to string and remove any non-numeric characters except decimal points and minus signs
                    cleaned = col_data.astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    # Convert to numeric
                    numeric_data = pd.to_numeric(cleaned, errors='coerce').fillna(0.0)
                    return numeric_data
                
                # Fallback: force conversion
                return pd.to_numeric(col_data.astype(str), errors='coerce').fillna(0.0)
                
            except Exception as e:
                st.warning(f"Error cleaning column {col_name}: {e}")
                # Return series of zeros as last resort
                return pd.Series([0.0] * len(chart_df), index=chart_df.index)
        
        # Clean Period column for x-axis
        try:
            if 'Period' in chart_df.columns:
                period_data = chart_df['Period']
                if period_data.dtype == 'object':
                    # Try to convert to datetime
                    try:
                        period_data = pd.to_datetime(period_data)
                    except:
                        # If that fails, try to clean first
                        period_data = pd.to_datetime(period_data.astype(str), errors='coerce')
                        # Fill any NaT with a default date
                        period_data = period_data.fillna(pd.to_datetime('2011-01-01'))
            else:
                # If no Period column, use the index
                period_data = chart_df.index
                if not isinstance(period_data, pd.DatetimeIndex):
                    period_data = pd.to_datetime(period_data, errors='coerce')
        except Exception as e:
            st.warning(f"Error processing Period data: {e}")
            # Create a default date range
            period_data = pd.date_range('2011-01-01', '2025-06-30', periods=len(chart_df))
        
        # Clean all cumulative return columns
        alpha_cum = clean_and_convert_column('Alpha_Cumulative')
        oni_cum = clean_and_convert_column('ONI_Cumulative')
        equal_cum = clean_and_convert_column('Equal_Cumulative')
        vni_cum = clean_and_convert_column('VNI_Cumulative')
        
        # Debug: Show data types and sample values
        st.info(f"Data types - Period: {type(period_data.iloc[0] if len(period_data) > 0 else 'empty')}, "
                f"Alpha: {type(alpha_cum.iloc[0] if len(alpha_cum) > 0 else 'empty')}")
        
        # Create the figure
        fig = go.Figure()
        
        # Add traces for each strategy using cleaned data
        try:
            fig.add_trace(go.Scatter(
                x=period_data,
                y=alpha_cum,
                mode='lines+markers',
                name='Alpha Strategy',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=4),
                hovertemplate='Date: %{x}<br>Alpha Strategy: %{y:.2f}%<extra></extra>'
            ))
        except Exception as e:
            st.error(f"Error adding Alpha trace: {e}")
        
        try:
            fig.add_trace(go.Scatter(
                x=period_data,
                y=oni_cum,
                mode='lines+markers',
                name='ONI Strategy',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=4),
                hovertemplate='Date: %{x}<br>ONI Strategy: %{y:.2f}%<extra></extra>'
            ))
        except Exception as e:
            st.error(f"Error adding ONI trace: {e}")
        
        try:
            fig.add_trace(go.Scatter(
                x=period_data,
                y=equal_cum,
                mode='lines+markers',
                name='Equal Weight',
                line=dict(color='#2ca02c', width=3),
                marker=dict(size=4),
                hovertemplate='Date: %{x}<br>Equal Weight: %{y:.2f}%<extra></extra>'
            ))
        except Exception as e:
            st.error(f"Error adding Equal Weight trace: {e}")
        
        try:
            fig.add_trace(go.Scatter(
                x=period_data,
                y=vni_cum,
                mode='lines+markers',
                name='VNI Benchmark',
                line=dict(color='#d62728', width=3),
                marker=dict(size=4),
                hovertemplate='Date: %{x}<br>VNI Benchmark: %{y:.2f}%<extra></extra>'
            ))
        except Exception as e:
            st.error(f"Error adding VNI trace: {e}")
        
        # Add timeline markers for Alpha strategy (with error handling)
        try:
            fig.add_vline(
                x=pd.to_datetime("2019-01-01"),
                line_dash="dash",
                line_color="gray",
                annotation_text="Gas/Coal Begin"
            )
            
            fig.add_vline(
                x=pd.to_datetime("2020-04-01"),
                line_dash="dash", 
                line_color="gray",
                annotation_text="Full Specialization"
            )
        except Exception as e:
            st.warning(f"Could not add timeline markers: {e}")
        
        # Update layout
        fig.update_layout(
            title='Power Sector Trading Strategies - Cumulative Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.success("✅ Chart created successfully")
        return fig
        
    except Exception as e:
        error_msg = f"Error creating unified strategy chart: {e}"
        st.error(error_msg)
        print(error_msg)
        import traceback
        st.error(f"Full traceback: {traceback.format_exc()}")
        return None
