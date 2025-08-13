import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import calendar
from plotly.subplots import make_subplots
import calendar
import io
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Power Sector Dashboard", layout="wide")

# Title
st.title("Power Sector Dashboard")

# Helper functions
@st.cache_data
def convert_df_to_excel(df, sheet_name="Data"):
    """Convert dataframe to Excel bytes for download"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

@st.cache_data  
def convert_df_to_csv(df):
    """Convert dataframe to CSV string for download"""
    return df.to_csv(index=False).encode('utf-8')

def add_download_buttons(df, filename_prefix, container=None):
    """Add download buttons for Excel and CSV"""
    if container is None:
        container = st
    
    col1, col2 = container.columns(2)
    
    with col1:
        excel_data = convert_df_to_excel(df)
        container.download_button(
            label="📊 Download as Excel",
            data=excel_data,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        csv_data = convert_df_to_csv(df)
        container.download_button(
            label="📄 Download as CSV", 
            data=csv_data,
            file_name=f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def calculate_ytd_growth(df, value_col, date_col, period_type):
    """Calculate proper YTD growth from cumulative values from beginning of year"""
    df = df.copy()
    df['Year'] = df[date_col].dt.year
    
    if period_type == "Monthly":
        df['Month'] = df[date_col].dt.month
        df = df.sort_values([date_col])
        
        # Calculate cumulative sum from beginning of each year
        df['Cumulative'] = df.groupby('Year')[value_col].cumsum()
        
        # For each month, get the cumulative value for same month in previous year
        df_pivot = df.pivot_table(index='Month', columns='Year', values='Cumulative', aggfunc='first')
        
        ytd_growth = []
        for _, row in df.iterrows():
            month = row['Month']
            year = row['Year']
            current_cumulative = row['Cumulative']
            
            # Get previous year's cumulative for same month
            if year-1 in df_pivot.columns and month in df_pivot.index:
                prev_year_cumulative = df_pivot.loc[month, year-1]
                if pd.notna(prev_year_cumulative) and prev_year_cumulative != 0:
                    growth = ((current_cumulative - prev_year_cumulative) / prev_year_cumulative) * 100
                    ytd_growth.append(growth)
                else:
                    ytd_growth.append(None)  # No previous year data
            else:
                ytd_growth.append(None)  # No previous year data
        
        return pd.Series(ytd_growth, index=df.index)
    
    elif period_type == "Quarterly":
        df['Quarter'] = df[date_col].dt.quarter
        df = df.sort_values([date_col])
        
        # For quarterly data, calculate cumulative within year
        df['Quarter_in_Year'] = df['Quarter']
        df['Cumulative'] = df.groupby(['Year', 'Quarter_in_Year'])[value_col].transform('first')
        
        # Calculate cumulative from Q1 to current quarter
        yearly_data = []
        for year in df['Year'].unique():
            year_df = df[df['Year'] == year].copy()
            year_df = year_df.sort_values('Quarter')
            year_df['Cumulative'] = year_df[value_col].cumsum()
            yearly_data.append(year_df)
        
        df = pd.concat(yearly_data).sort_values([date_col])
        
        # Compare with same quarter cumulative in previous year
        df_pivot = df.pivot_table(index='Quarter', columns='Year', values='Cumulative', aggfunc='first')
        
        ytd_growth = []
        for _, row in df.iterrows():
            quarter = row['Quarter']
            year = row['Year']
            current_cumulative = row['Cumulative']
            
            if year-1 in df_pivot.columns and quarter in df_pivot.index:
                prev_year_cumulative = df_pivot.loc[quarter, year-1]
                if pd.notna(prev_year_cumulative) and prev_year_cumulative != 0:
                    growth = ((current_cumulative - prev_year_cumulative) / prev_year_cumulative) * 100
                    ytd_growth.append(growth)
                else:
                    ytd_growth.append(None)
            else:
                ytd_growth.append(None)
        
        return pd.Series(ytd_growth, index=df.index)
    
    else:
        # For semi-annual and annual, YTD doesn't make much sense, return simple growth
        return df[value_col].pct_change() * 100

def calculate_yoy_growth(df, value_col, periods):
    """Calculate YoY growth only when sufficient historical data exists"""
    growth = df[value_col].pct_change(periods=periods) * 100
    
    # Set growth to NaN for periods where we don't have enough historical data
    if len(df) > periods:
        growth.iloc[:periods] = None
    else:
        growth[:] = None
        
    return growth

def update_chart_layout_with_no_secondary_grid(fig):
    """Remove gridlines from secondary y-axis while keeping the axis"""
    fig.update_layout(
        yaxis2=dict(
            showgrid=False,  # Remove secondary y-axis gridlines
            zeroline=False   # Remove zero line for secondary axis
        )
    )
    return fig

def calculate_power_rating(df, power_col, date_col):
    """Calculate rating based on latest complete quarter's QoQ and YoY growth"""
    try:
        # Get current date info
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        current_quarter = (current_month - 1) // 3 + 1
        
        # Get latest quarter data
        df_temp = df.copy()
        df_temp['Year'] = df_temp[date_col].dt.year
        df_temp['Quarter'] = df_temp[date_col].dt.quarter
        df_temp['Month'] = df_temp[date_col].dt.month
        
        # Group by quarter and get quarterly sums
        quarterly_df = df_temp.groupby(['Year', 'Quarter'])[power_col].sum().reset_index()
        quarterly_df['Date'] = pd.to_datetime([f"{y}-{q*3:02d}-01" for y, q in zip(quarterly_df['Year'], quarterly_df['Quarter'])])
        quarterly_df = quarterly_df.sort_values('Date')
        
        if len(quarterly_df) < 2:
            return "Neutral", "Insufficient data"
        
        # Determine which quarter to use for rating
        latest_quarter_row = quarterly_df.iloc[-1]
        latest_year = latest_quarter_row['Year']
        latest_quarter_num = latest_quarter_row['Quarter']
        
        # Check if current quarter is incomplete (use preceding quarter if so)
        if latest_year == current_year and latest_quarter_num == current_quarter:
            current_quarter_months = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9], 4: [10, 11, 12]}
            required_months = current_quarter_months[current_quarter]
            quarter_data = df_temp[(df_temp['Year'] == current_year) & (df_temp['Quarter'] == current_quarter)]
            available_months = quarter_data['Month'].unique()
            if len(available_months) < len(required_months):
                if len(quarterly_df) >= 2:
                    rating_quarter = quarterly_df.iloc[-2]  # Use preceding quarter
                else:
                    return "Neutral", "Insufficient complete quarter data"
            else:
                rating_quarter = latest_quarter_row  # Current quarter is complete
        else:
            rating_quarter = latest_quarter_row  # Latest quarter is from previous period
        
        # Calculate QoQ growth (quarter over quarter)
        quarterly_df['QoQ_Growth'] = quarterly_df[power_col].pct_change() * 100
        
        # Find the rating quarter in the dataframe
        rating_quarter_idx = quarterly_df[
            (quarterly_df['Year'] == rating_quarter['Year']) & 
            (quarterly_df['Quarter'] == rating_quarter['Quarter'])
        ].index[0]
        
        qoq_growth = quarterly_df.loc[rating_quarter_idx, 'QoQ_Growth']
        
        # For renewable, use only QoQ growth for rating
        if power_col == 'Total_Renewable':
            if pd.notna(qoq_growth):
                if qoq_growth > 5:
                    return "Positive", f"QoQ: {qoq_growth:.1f}%"
                elif qoq_growth < 5:
                    return "Negative", f"QoQ: {qoq_growth:.1f}%"
                else:
                    return "Neutral", f"QoQ: {qoq_growth:.1f}%"
            else:
                return "Neutral", "Insufficient data"
        # For other power types, keep old logic
        else:
            # Calculate YoY growth (year over year, 4 quarters back)
            quarterly_df['YoY_Growth'] = quarterly_df[power_col].pct_change(periods=4) * 100
            yoy_growth = quarterly_df.loc[rating_quarter_idx, 'YoY_Growth']
            if pd.notna(qoq_growth) and pd.notna(yoy_growth):
                if qoq_growth > 5 and yoy_growth > 5:
                    return "Positive", f"QoQ: {qoq_growth:.1f}%, YoY: {yoy_growth:.1f}%"
                elif qoq_growth < 5 and yoy_growth < 5:
                    return "Negative", f"QoQ: {qoq_growth:.1f}%, YoY: {yoy_growth:.1f}%"
                else:
                    return "Neutral", f"QoQ: {qoq_growth:.1f}%, YoY: {yoy_growth:.1f}%"
            elif pd.notna(qoq_growth):
                if qoq_growth > 5:
                    return "Positive", f"QoQ: {qoq_growth:.1f}%, YoY: N/A"
                elif qoq_growth < 5:
                    return "Negative", f"QoQ: {qoq_growth:.1f}%, YoY: N/A"
                else:
                    return "Neutral", f"QoQ: {qoq_growth:.1f}%, YoY: N/A"
            else:
                return "Neutral", "Insufficient data"
    except Exception as e:
        return "Neutral", f"Error calculating rating: {str(e)}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def create_stock_performance_chart(stock_symbols, sector_name):
    """Create a stock price performance chart for year-to-date using real data"""
    try:
        import yfinance as yf
        import pandas as pd
        from datetime import datetime, date
        
        # Get current year start date
        current_year = datetime.now().year
        year_start = f"{current_year}-01-01"
        today = datetime.now().strftime("%Y-%m-%d")
        
        stocks_data = {}
        
        # Add a small delay to avoid rate limiting
        import time
        
    # Do NOT fetch data; only fetch sector stocks
        
        # Fetch individual stock data
        for symbol in stock_symbols:
            try:
                # Add .VN suffix for Vietnamese stocks
                ticker_symbol = f"{symbol}.VN"
                stock = yf.download(ticker_symbol, start=year_start, end=today, progress=False, auto_adjust=True)
                
                if len(stock) > 1 and 'Close' in stock.columns:
                    stock_start = float(stock['Close'].iloc[0])
                    stock_current = float(stock['Close'].iloc[-1])
                    if stock_start > 0:  # Avoid division by zero
                        ytd_performance = ((stock_current - stock_start) / stock_start) * 100
                        stocks_data[symbol] = float(ytd_performance)
                    else:
                        stocks_data[symbol] = 0.0
                else:
                    # If no data found, try without .VN suffix
                    time.sleep(0.1)  # Small delay to avoid rate limiting
                    stock = yf.download(symbol, start=year_start, end=today, progress=False, auto_adjust=True)
                    if len(stock) > 1 and 'Close' in stock.columns:
                        stock_start = float(stock['Close'].iloc[0])
                        stock_current = float(stock['Close'].iloc[-1])
                        if stock_start > 0:  # Avoid division by zero
                            ytd_performance = ((stock_current - stock_start) / stock_start) * 100
                            stocks_data[symbol] = float(ytd_performance)
                        else:
                            stocks_data[symbol] = 0.0
                    else:
                        stocks_data[symbol] = 0.0  # Default to 0 if no data
            except Exception as e:
                stocks_data[symbol] = 0.0  # Default to 0 if error
                time.sleep(0.1)  # Small delay before next request
        
        # Create bar chart
        symbols = list(stocks_data.keys())
        performances = list(stocks_data.values())
        
        # Ensure we have data
        if not symbols or not performances:
            fig = go.Figure()
            fig.add_annotation(
                text="No stock data available",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14)
            )
            fig.update_layout(title=f"{sector_name} Stocks - Year-to-Date Performance", height=400)
            return fig
        
        # Color coding: green for positive, red for negative
        colors = ['green' if p >= 0 else 'red' for p in performances]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=performances,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in performances],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=f"{sector_name} Stocks - Year-to-Date Performance ({current_year})",
            xaxis_title="Stock Symbol",
            yaxis_title="YTD Performance (%)",
            height=400,
            showlegend=False
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        # Return error figure if something goes wrong
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading stock data: {str(e)}<br>Please check internet connection and try again", 
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title=f"{sector_name} Stocks - Year-to-Date Performance",
            height=400
        )
        return fig

# Load and process data
@st.cache_data
def load_data():
    # Load monthly power volume data
    df = pd.read_csv('power_volume.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Clean numeric columns - remove commas and spaces, convert to numeric
    numeric_columns = ['Hydro', 'Coals', 'Gas', 'Renewables', 'Import & Diesel']
    for col in numeric_columns:
        if col in df.columns:
            # Remove spaces and commas, then convert to numeric
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Half'] = (df['Date'].dt.month - 1) // 6 + 1
    
    # Load renewable data
    renewable_df = None
    has_renewable_data = False
    try:
        renewable_df = pd.read_csv('thong_so_vh.csv')
        renewable_df['date'] = pd.to_datetime(renewable_df['date'], format='%d/%m/%Y')
        renewable_df['Year'] = renewable_df['date'].dt.year
        renewable_df['Month'] = renewable_df['date'].dt.month
        renewable_df['Quarter'] = renewable_df['date'].dt.quarter
        renewable_df['Half'] = (renewable_df['date'].dt.month - 1) // 6 + 1
        has_renewable_data = True
    except FileNotFoundError:
        st.warning("Renewable data file 'thong_so_vh.csv' not found.")
    
    # Load weighted average price data (CGM Price)
    cgm_df = None
    has_cgm_data = False
    try:
        cgm_df = pd.read_csv('weighted_average_prices.csv')
        cgm_df['date'] = pd.to_datetime(cgm_df['date'])
        cgm_df['Year'] = cgm_df['date'].dt.year
        cgm_df['Month'] = cgm_df['date'].dt.month
        cgm_df['Quarter'] = cgm_df['date'].dt.quarter
        cgm_df['Half'] = (cgm_df['date'].dt.month - 1) // 6 + 1
        has_cgm_data = True
    except FileNotFoundError:
        st.warning("CGM price data file 'weighted_average_prices.csv' not found.")
    
    # Load thermal data
    thermal_df = None
    has_thermal_data = False
    try:
        thermal_df = pd.read_excel('thermal_price.xlsx')
        
        # Try to find date column with different possible names and check first column
        date_col = None
        
        # Check first column first (most likely to be dates)
        if len(thermal_df.columns) > 0:
            first_col = thermal_df.columns[0]
            try:
                # Test if first column can be converted to datetime
                test_dates = pd.to_datetime(thermal_df[first_col])
                date_col = first_col
            except:
                pass
        
        # If first column isn't dates, check for date-related column names
        if date_col is None:
            for col in thermal_df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'ngay' in col.lower():
                    try:
                        test_dates = pd.to_datetime(thermal_df[col])
                        date_col = col
                        break
                    except:
                        continue
        
        if date_col:
            thermal_df['Date'] = pd.to_datetime(thermal_df[date_col])
            thermal_df['Year'] = thermal_df['Date'].dt.year
            thermal_df['Month'] = thermal_df['Date'].dt.month
            thermal_df['Quarter'] = thermal_df['Date'].dt.quarter
            thermal_df['Half'] = (thermal_df['Date'].dt.month - 1) // 6 + 1
            
            # Filter to only include data up to current month of current year
            current_date = pd.Timestamp.now()
            thermal_df = thermal_df[thermal_df['Date'] <= current_date]
        else:
            st.warning("No valid date column found in thermal data. Please check the file format.")
            thermal_df = None
        
        has_thermal_data = True
    except FileNotFoundError:
        st.warning("Thermal data file 'thermal_price.xlsx' not found.")
    
    # Load reservoir data
    reservoir_df = None
    has_reservoir_data = False
    try:
        reservoir_df = pd.read_csv('water_reservoir.csv')
        # Fix date parsing with correct format
        reservoir_df['date_time'] = pd.to_datetime(reservoir_df['date_time'], format='%d/%m/%Y %H:%M')
        has_reservoir_data = True
    except FileNotFoundError:
        st.warning("Reservoir data file 'water_reservoir.csv' not found.")
    except Exception as e:
        st.warning(f"Error loading reservoir data: {e}")
        reservoir_df = None
        has_reservoir_data = False
    
    return df, renewable_df, thermal_df, cgm_df, reservoir_df, has_renewable_data, has_thermal_data, has_cgm_data, has_reservoir_data

# Load all data
df, renewable_df, thermal_df, cgm_df, reservoir_df, has_renewable_data, has_thermal_data, has_cgm_data, has_reservoir_data = load_data()

# Create tabs - clean tab structure
tab_list = ["⚡Power Volume", "💲CGM Price"]
if has_renewable_data:
    tab_list.append("🍃Renewable Power")
tab_list.extend(["💧Hydro Power", "🪨Coal-fired Power", "🔥Gas-fired Power"])

tabs = st.tabs(tab_list)

# Tab 1: Power Volume
with tabs[0]:
    st.header("Power Volume Analysis")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        period = st.selectbox(
            "Select time period:",
            ["Monthly", "Quarterly", "Semi-annually", "Annually"],
            index=0,
            key="power_volume_period"
        )
    
    # Remove YoY from growth type selection
    with col2:
        growth_type = st.selectbox(
            "Select growth type:",
            ["Year-to-Date (YTD)"],
            index=0,
            key="power_volume_growth"
        )
    
    with col3:
        selected_power_types = st.multiselect(
            "Select power types:",
            ["Gas", "Hydro", "Coals", "Renewables", "Import & Diesel"],
            default=["Gas", "Hydro", "Coals", "Renewables", "Import & Diesel"],
            key="power_types_selection"
        )
    
    # Filter data based on period
    if period == "Monthly":
        filtered_df = df[['Date', 'Gas', 'Hydro', 'Coals', 'Renewables', 'Import & Diesel']].copy()
    elif period == "Quarterly":
        filtered_df = df.groupby(['Year', 'Quarter'])[['Gas', 'Hydro', 'Coals', 'Renewables', 'Import & Diesel']].sum().reset_index()
        filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3}-01" for y, q in zip(filtered_df['Year'], filtered_df['Quarter'])])
    elif period == "Semi-annually":
        filtered_df = df.groupby(['Year', 'Half'])[['Gas', 'Hydro', 'Coals', 'Renewables', 'Import & Diesel']].sum().reset_index()
        filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])
    else:  # Annually
        filtered_df = df.groupby('Year')[['Gas', 'Hydro', 'Coals', 'Renewables', 'Import & Diesel']].sum().reset_index()
        filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in filtered_df['Year']])
    
    # Calculate total and growth for selected power types only
    filtered_df['Total'] = filtered_df[selected_power_types].sum(axis=1)
    
    # Ensure Total column is numeric and handle NaN values
    filtered_df['Total'] = pd.to_numeric(filtered_df['Total'], errors='coerce')
    
    # Improved growth calculations
    if growth_type == "Year-over-Year (YoY)":
        periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
        filtered_df['Total_Growth'] = calculate_yoy_growth(filtered_df, 'Total', periods_map[period])
        growth_title = "YoY Growth (%)"
    else:
        filtered_df['Total_Growth'] = calculate_ytd_growth(filtered_df, 'Total', 'Date', period)
        growth_title = "YTD Growth (%)"
    
    # Create chart with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add stacked bars for selected power types only
    power_types = ['Gas', 'Hydro', 'Coals', 'Renewables', 'Import & Diesel']
    power_names = ['Gas Power', 'Hydro Power', 'Coal Power', 'Renewables', 'Import & Diesel']
    colors = ['#0C4130', '#08C179', '#B78D51', '#C0C1C2', '#97999B']
    
    # Create x-axis labels and growth calculation based on period
    if period == "Monthly":
        x_labels = [d.strftime('%b %Y') for d in filtered_df['Date']]
        filtered_df['Growth'] = filtered_df['Total'].pct_change() * 100  # MoM
        growth_title = "MoM Growth (%)"
    elif period == "Quarterly":
        x_labels = [f"Q{d.quarter} {d.year}" for d in filtered_df['Date']]
        filtered_df['Growth'] = filtered_df['Total'].pct_change() * 100  # QoQ
        growth_title = "QoQ Growth (%)"
    elif period == "Semi-annually":
        x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in filtered_df['Date']]
        filtered_df['Growth'] = filtered_df['Total'].pct_change() * 100  # HoH
        growth_title = "HoH Growth (%)"
    else:
        x_labels = [str(int(d.year)) for d in filtered_df['Date']]
        filtered_df['Growth'] = filtered_df['Total'].pct_change() * 100  # YoY
        growth_title = "YoY Growth (%)"
    
    # Add stacked bars for each selected power type
    for i, (power_type, power_name) in enumerate(zip(power_types, power_names)):
        if power_type in selected_power_types:
            fig.add_trace(
                go.Bar(
                    name=power_name,
                    x=x_labels,
                    y=filtered_df[power_type],
                    marker_color=colors[i],
                    hovertemplate=f"{power_name}<br>%{{x}}<br>Volume: %{{y}} MWh<extra></extra>"
                ),
                secondary_y=False
            )
    
    # Add growth line
    fig.add_trace(
        go.Scatter(
            name=f"Total {growth_title}",
            x=x_labels,
            y=filtered_df['Total_Growth'],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=4),
            hovertemplate=f"Total {growth_title}<br>%{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f'{period} Power Volume with {growth_title}',
        barmode='stack',
        hovermode='x unified',
        showlegend=True
    )
    
    fig.update_yaxes(title_text="Volume (MWh)", secondary_y=False)
    fig.update_yaxes(title_text=growth_title, secondary_y=True)
    fig.update_xaxes(title_text="Date")
    
    # Remove secondary y-axis gridlines
    fig = update_chart_layout_with_no_secondary_grid(fig)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download data section
    st.subheader("📥 Download Data")
    download_df = filtered_df[['Date'] + selected_power_types + ['Total', 'Total_Growth']].copy()
    download_df['Period_Label'] = x_labels
    add_download_buttons(download_df, f"power_volume_{period.lower()}_{growth_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}")

# Tab 2: CGM Price
with tabs[1]:
    st.header("CGM Price Analysis")
    
    if has_cgm_data and cgm_df is not None:
        # Controls
        cgm_col1, cgm_col2 = st.columns(2)
        
        with cgm_col1:
            cgm_period = st.selectbox(
                "Select time period:",
                ["Monthly", "Quarterly", "Semi-annually", "Annually"],
                index=0,
                key="cgm_period"
            )
        
        # Filter out future years (2026 and beyond)
        current_year = pd.Timestamp.now().year
        cgm_df_filtered = cgm_df[cgm_df['Year'] <= current_year]
        
        # Filter and aggregate data
        if cgm_period == "Monthly":
            cgm_filtered_df = cgm_df_filtered.groupby(['Year', 'Month'])['weighted_avg_price'].mean().reset_index()
            cgm_filtered_df['Month_Name'] = cgm_filtered_df['Month'].apply(lambda x: calendar.month_abbr[x])
            cgm_filtered_df['Period'] = cgm_filtered_df['Month_Name']
        elif cgm_period == "Quarterly":
            cgm_filtered_df = cgm_df_filtered.groupby(['Year', 'Quarter'])['weighted_avg_price'].mean().reset_index()
            cgm_filtered_df['Period'] = cgm_filtered_df['Quarter'].apply(lambda x: f"Q{x}")
        elif cgm_period == "Semi-annually":
            cgm_filtered_df = cgm_df_filtered.groupby(['Year', 'Half'])['weighted_avg_price'].mean().reset_index()
            cgm_filtered_df['Period'] = cgm_filtered_df['Half'].apply(lambda x: f"H{x}")
        else:  # Annually
            cgm_filtered_df = cgm_df_filtered.groupby('Year')['weighted_avg_price'].mean().reset_index()
            cgm_filtered_df['Period'] = cgm_filtered_df['Year'].astype(str)
        
        # Create chart with separate lines/bars for each year
        cgm_fig = go.Figure()
        
        years = sorted(cgm_filtered_df['Year'].unique())
        colors = ['#0C4130', '#08C179', '#C0C1C2', '#97999B', '#B78D51', '#014ABD']

        if cgm_period == "Annually":
            # Use bar chart for annual data
            cgm_fig.add_trace(
                go.Bar(
                    name="CGM Price",
                    x=[str(int(year)) for year in years],
                    y=[cgm_filtered_df[cgm_filtered_df['Year'] == year]['weighted_avg_price'].iloc[0] for year in years],
                    marker_color='#08C179',
                    hovertemplate="Year: %{x}<br>CGM Price: %{y:,.2f} VND/kWh<extra></extra>"
                )
            )
        else:
            # Use line chart for other periods
            for i, year in enumerate(years):
                year_data = cgm_filtered_df[cgm_filtered_df['Year'] == year]
                
                cgm_fig.add_trace(
                    go.Scatter(
                        name=str(year),
                        x=year_data['Period'],
                        y=year_data['weighted_avg_price'],
                        mode='lines+markers',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=6),
                        hovertemplate=f"Year: {year}<br>Period: %{{x}}<br>CGM Price: %{{y:,.2f}} VND/kWh<extra></extra>"
                    )
                )
        
        cgm_fig.update_layout(
            title=f"{cgm_period} CGM Price {'Analysis' if cgm_period == 'Annually' else 'Trend'}",
            xaxis_title="Year" if cgm_period == "Annually" else "Time Period",
            yaxis_title="CGM Price (VND/kWh)",
            hovermode='x unified' if cgm_period != "Annually" else 'closest'
        )
        
        st.plotly_chart(cgm_fig, use_container_width=True)
        
        # Download data section
        st.subheader("📥 Download Data")
        add_download_buttons(cgm_filtered_df, f"cgm_price_{cgm_period.lower()}")
    else:
        st.error("CGM price data not available.")

# Tab 3: Renewable Power (if available)
if has_renewable_data:
    with tabs[2]:
        # Calculate rating for renewable power
        renewable_total_col = ['dien_gio_mkWh', 'dmt_trang_trai_mkWh', 'dmt_mai_thuong_pham_mkWh']
        renewable_df_rating = renewable_df[renewable_total_col + ['date']].copy()
        renewable_df_rating['Total_Renewable'] = renewable_df_rating[renewable_total_col].sum(axis=1)
        rating, rating_details = calculate_power_rating(renewable_df_rating, 'Total_Renewable', 'date')
        
        # Header with rating
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("Renewable Power Analysis")
        with col2:
            if rating == "Positive":
                st.success(f"**Rating: {rating}**")
            elif rating == "Negative":
                st.error(f"**Rating: {rating}**")
            else:
                st.info(f"**Rating: {rating}**")
            st.caption(rating_details)
        
        # Controls
        ren_col1, ren_col2, ren_col3 = st.columns(3)
        
        with ren_col1:
            ren_period = st.selectbox(
                "Select time period:",
                ["Monthly", "Quarterly", "Semi-annually", "Annually"],
                index=0,
                key="renewable_period"
            )
        
        with ren_col2:
            selected_renewable_types = st.multiselect(
                "Select renewable types:",
                ["Wind Power", "Solar Farm", "Rooftop Solar"],
                default=["Wind Power", "Solar Farm", "Rooftop Solar"],
                key="renewable_types_selection"
            )
        
        with ren_col3:
            if ren_period == "Monthly":
                growth_options = ["MoM"]
            elif ren_period == "Quarterly":
                growth_options = ["QoQ"]
            elif ren_period == "Semi-annually":
                growth_options = ["HoH"]
            else:
                growth_options = ["YoY"]
            ren_growth_type = st.selectbox(
                "Select growth type:",
                growth_options,
                index=0,
                key="renewable_growth_type"
            )
        
        # Map display names to column names
        renewable_cols_map = {
            "Wind Power": "dien_gio_mkWh",
            "Solar Farm": "dmt_trang_trai_mkWh", 
            "Rooftop Solar": "dmt_mai_thuong_pham_mkWh"
        }
        
        # Get selected columns
        selected_renewable_cols = [renewable_cols_map[name] for name in selected_renewable_types]
        
        # Filter data - aggregate to monthly and sum daily values
        all_renewable_cols = ['dien_gio_mkWh', 'dmt_trang_trai_mkWh', 'dmt_mai_thuong_pham_mkWh']
        
        if ren_period == "Monthly":
            # Sum daily data to monthly
            ren_filtered_df = renewable_df.groupby(['Year', 'Month'])[all_renewable_cols].sum().reset_index()
            ren_filtered_df['Date'] = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(ren_filtered_df['Year'], ren_filtered_df['Month'])])
        elif ren_period == "Quarterly":
            ren_filtered_df = renewable_df.groupby(['Year', 'Quarter'])[all_renewable_cols].sum().reset_index()
            ren_filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3}-01" for y, q in zip(ren_filtered_df['Year'], ren_filtered_df['Quarter'])])
        elif ren_period == "Semi-annually":
            ren_filtered_df = renewable_df.groupby(['Year', 'Half'])[all_renewable_cols].sum().reset_index()
            ren_filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6}-01" for y, h in zip(ren_filtered_df['Year'], ren_filtered_df['Half'])])
        else:  # Annually
            ren_filtered_df = renewable_df.groupby('Year')[all_renewable_cols].sum().reset_index()
            ren_filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in ren_filtered_df['Year']])
        
        # Calculate total renewable volume for selected types only
        ren_filtered_df['Total_Renewable'] = ren_filtered_df[selected_renewable_cols].sum(axis=1)
        
        # Calculate growth
        if ren_growth_type == "MoM":
            ren_filtered_df['Total_Growth'] = ren_filtered_df['Total_Renewable'].pct_change() * 100
            growth_title = "MoM Growth (%)"
        elif ren_growth_type == "QoQ":
            ren_filtered_df['Total_Growth'] = ren_filtered_df['Total_Renewable'].pct_change() * 100
            growth_title = "QoQ Growth (%)"
        elif ren_growth_type == "HoH":
            ren_filtered_df['Total_Growth'] = ren_filtered_df['Total_Renewable'].pct_change() * 100
            growth_title = "HoH Growth (%)"
        else:  # YoY
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            ren_filtered_df['Total_Growth'] = calculate_yoy_growth(ren_filtered_df, 'Total_Renewable', periods_map[ren_period])
            growth_title = "YoY Growth (%)"
        
        # Create chart with secondary y-axis
        ren_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add stacked bars for selected types only
        ren_names_map = {
            "dien_gio_mkWh": "Wind Power",
            "dmt_trang_trai_mkWh": "Solar Farm", 
            "dmt_mai_thuong_pham_mkWh": "Rooftop Solar"
        }
        ren_colors = ['#08C179', '#B78D51', '#C0C1C2']
        
        # Create x-axis labels based on period
        if ren_period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in ren_filtered_df['Date']]
        elif ren_period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in ren_filtered_df['Date']]
        elif ren_period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in ren_filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in ren_filtered_df['Date']]
        
        color_idx = 0
        for col in selected_renewable_cols:
            ren_fig.add_trace(
                go.Bar(
                    name=ren_names_map[col],
                    x=x_labels,
                    y=ren_filtered_df[col],
                    marker_color=ren_colors[color_idx % len(ren_colors)],
                    hovertemplate=f"{ren_names_map[col]}<br>%{{x}}<br>Volume: %{{y}} MkWh<extra></extra>"
                ),
                secondary_y=False
            )
            color_idx += 1
        
        # Add growth line
        ren_fig.add_trace(
            go.Scatter(
                name=f"Total {growth_title}",
                x=x_labels,
                y=ren_filtered_df['Total_Growth'],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                hovertemplate=f"Total {growth_title}<br>%{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>"
            ),
            secondary_y=True
        )
        
        # Update layout
        ren_fig.update_layout(
            title=f'{ren_period} Renewable Power Volume with {growth_title}',
            barmode='stack',
            hovermode='x unified',
            showlegend=True
        )
        
        ren_fig.update_yaxes(title_text="Volume (MkWh)", secondary_y=False)
        ren_fig.update_yaxes(title_text=growth_title, secondary_y=True)
        ren_fig.update_xaxes(title_text="Date")
        
        # Remove secondary y-axis gridlines
        ren_fig = update_chart_layout_with_no_secondary_grid(ren_fig)
        
        st.plotly_chart(ren_fig, use_container_width=True)
        
        # Download data section
        st.subheader("📥 Download Data")
        ren_download_df = ren_filtered_df[['Date'] + selected_renewable_cols + ['Total_Renewable', 'Total_Growth']].copy()
        ren_download_df['Period_Label'] = x_labels
        add_download_buttons(ren_download_df, f"renewable_power_{ren_period.lower()}_{ren_growth_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}")

        # Stock Performance Chart
        st.subheader("📈 Renewable Sector Stocks Performance (YTD)")
        renewable_stocks = ['REE', 'PC1', 'HDG', 'GEG', 'GHC', 'BCG']
        renewable_stock_fig = create_stock_performance_chart(renewable_stocks, "Renewable Power")
        st.plotly_chart(renewable_stock_fig, use_container_width=True)

# Tab 4: Hydro Power
hydro_tab_index = 3 if has_renewable_data else 2
with tabs[hydro_tab_index]:
    # Calculate rating for hydro power
    hydro_power_df_rating = df[['Hydro', 'Date']].copy()
    hydro_power_df_rating = hydro_power_df_rating.rename(columns={'Date': 'date'})
    rating, rating_details = calculate_power_rating(hydro_power_df_rating, 'Hydro', 'date')
    
    # Header with rating
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Hydro Power Analysis")
    with col2:
        if rating == "Positive":
            st.success(f"**Rating: {rating}**")
        elif rating == "Negative":
            st.error(f"**Rating: {rating}**")
        else:
            st.info(f"**Rating: {rating}**")
        st.caption(rating_details)
    
    # Controls
    hydro_col1, hydro_col2 = st.columns(2)
    
    with hydro_col1:
        hydro_chart_type = st.selectbox(
            "Select Chart Type:",
            ["Total Capacity Comparison", "Hydro Power Volume"],
            key="hydro_chart_type"
        )
    
    with hydro_col2:
        # Leave this column empty for both chart types
        # as controls are shown in the appropriate sections below
        pass
    
    if hydro_chart_type == "Total Capacity Comparison" and has_reservoir_data:
        st.subheader("Total Capacity Comparison (2024 vs 2025)")
        
        # Create region translation dictionary
        region_translation = {
            "Đông Bắc Bộ": "Northeast",
            "Tây Bắc Bộ": "Northwest", 
            "Bắc Trung Bộ": "North Central",
            "Nam Trung Bộ": "South Central",
            "Tây Nguyên": "Central Highlands"
        }
        
        # Add English region names
        reservoir_df['region_en'] = reservoir_df['region'].map(region_translation)
        
        # Region selector with English names
        hydro_region = st.selectbox(
            "Select Region:",
            sorted(reservoir_df['region_en'].dropna().unique()),
            key="hydro_region_capacity"
        )
        # Growth type selector for average capacity
        hydro_capacity_growth_type = st.selectbox(
            "Select growth type:",
            ["MoM", "YTD"],
            index=0,
            key="hydro_capacity_growth_type"
        )
        
        # Process reservoir data
        reservoir_df['Year'] = reservoir_df['date_time'].dt.year
        reservoir_df['Month'] = reservoir_df['date_time'].dt.month
        
        # Filter for selected region and years 2024, 2025
        region_data = reservoir_df[(reservoir_df['region_en'] == hydro_region) & 
                                 (reservoir_df['Year'].isin([2024, 2025]))].copy()
        
        # Convert total_capacity to numeric
        region_data['total_capacity'] = pd.to_numeric(region_data['total_capacity'], errors='coerce')
        
        # Aggregate by year and month (average capacity per month)
        capacity_comparison = region_data.groupby(['Year', 'Month'])['total_capacity'].mean().reset_index()
        # Add a proper date column for YTD growth calculation
        capacity_comparison['Date'] = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(capacity_comparison['Year'], capacity_comparison['Month'])])
        
        # Calculate growth
        if hydro_capacity_growth_type == "MoM":
            capacity_comparison['Growth'] = capacity_comparison['total_capacity'].pct_change() * 100
            growth_title = "MoM Growth (%)"
        else:  # YTD
            capacity_comparison['Growth'] = calculate_ytd_growth(capacity_comparison, 'total_capacity', 'Date', 'Monthly')
            growth_title = "YTD Growth (%)"
        # Create comparison chart with secondary y-axis for growth
        capacity_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bars for each year
        for year in [2024, 2025]:
            year_data = capacity_comparison[capacity_comparison['Year'] == year]
            if len(year_data) > 0:
                month_labels = [calendar.month_abbr[m] for m in year_data['Month']]
                
                capacity_fig.add_trace(
                    go.Bar(
                        name=f"{year}",
                        x=month_labels,
                        y=year_data['total_capacity'],
                        marker_color='#A9A9A9' if year == 2024 else '#08C179',
                        hovertemplate=f"{year}<br>Month: %{{x}}<br>Avg Capacity: %{{y:.1f}} Million m³<extra></extra>"
                    ),
                    secondary_y=False
                )
        
        # Calculate and add YoY growth line
        if len(capacity_comparison[capacity_comparison['Year'] == 2025]) > 0 and len(capacity_comparison[capacity_comparison['Year'] == 2024]) > 0:
            # Merge 2024 and 2025 data to calculate YoY growth
            capacity_2024 = capacity_comparison[capacity_comparison['Year'] == 2024].set_index('Month')['total_capacity']
            capacity_2025 = capacity_comparison[capacity_comparison['Year'] == 2025].set_index('Month')['total_capacity']
            
            # Calculate YoY growth for months that exist in both years
            common_months = capacity_2024.index.intersection(capacity_2025.index)
            if len(common_months) > 0:
                yoy_growth = []
                month_labels_growth = []
                for month in sorted(common_months):
                    if capacity_2024[month] > 0:
                        growth = ((capacity_2025[month] - capacity_2024[month]) / capacity_2024[month]) * 100
                        yoy_growth.append(growth)
                        month_labels_growth.append(calendar.month_abbr[month])
                
                if yoy_growth:
                    capacity_fig.add_trace(
                        go.Scatter(
                            name="YoY Growth (%)",
                            x=month_labels_growth,
                            y=yoy_growth,
                            mode='lines+markers',
                            line=dict(color='red', width=2),
                            marker=dict(size=4),
                            hovertemplate="YoY Growth<br>Month: %{x}<br>Growth: %{y:.2f}%<extra></extra>"
                        ),
                        secondary_y=True
                    )
        
        capacity_fig.update_layout(
            title=f"Monthly Total Capacity Comparison (2024 vs 2025) with YoY Growth - {hydro_region}",
            xaxis_title="Month",
            yaxis_title="Average Total Capacity (Million m³)",
            hovermode='x unified',
            showlegend=True,
            barmode='group'
        )
        
        capacity_fig.update_yaxes(title_text="Average Total Capacity (Million m³)", secondary_y=False)
        capacity_fig.update_yaxes(title_text="YoY Growth (%)", secondary_y=True)
        
        # Remove secondary y-axis gridlines
        capacity_fig = update_chart_layout_with_no_secondary_grid(capacity_fig)
        
        st.plotly_chart(capacity_fig, use_container_width=True)
        
        # Download data section
        st.subheader("📥 Download Data")
        add_download_buttons(capacity_comparison, f"water_capacity_{hydro_region.lower().replace(' ', '_')}")
        
    else:  # Hydro Power Volume
        st.subheader("Hydro Power Volume")
        
        # Add period and growth type selectors for volume analysis
        volume_col1, volume_col2 = st.columns(2)
        
        with volume_col1:
            hydro_period = st.selectbox(
                "Select Time Period:",
                ["Monthly", "Quarterly", "Semi-annually", "Annually"],
                key="hydro_volume_period"
            )
        
        with volume_col2:
            # Growth type selector
            hydro_growth_type = st.selectbox(
                "Select Growth Type:",
                ["Year-over-Year (YoY)", "Year-to-Date (YTD)"],
                key="hydro_growth_type"
            )
        
        # Filter hydro power data
        if hydro_period == "Monthly":
            hydro_filtered_df = df[['Date', 'Hydro']].copy()
        elif hydro_period == "Quarterly":
            hydro_filtered_df = df.groupby(['Year', 'Quarter'])['Hydro'].sum().reset_index()
            hydro_filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3}-01" for y, q in zip(hydro_filtered_df['Year'], hydro_filtered_df['Quarter'])])
        elif hydro_period == "Semi-annually":
            hydro_filtered_df = df.groupby(['Year', 'Half'])['Hydro'].sum().reset_index()
            hydro_filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6}-01" for y, h in zip(hydro_filtered_df['Year'], hydro_filtered_df['Half'])])
        else:  # Annually
            hydro_filtered_df = df.groupby('Year')['Hydro'].sum().reset_index()
            hydro_filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in hydro_filtered_df['Year']])
        
        # Calculate growth
        if hydro_growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            hydro_filtered_df['Growth'] = calculate_yoy_growth(hydro_filtered_df, 'Hydro', periods_map[hydro_period])
            growth_title = "YoY Growth (%)"
        else:
            hydro_filtered_df['Growth'] = calculate_ytd_growth(hydro_filtered_df, 'Hydro', 'Date', hydro_period)
            growth_title = "YTD Growth (%)"
        
        # Create chart with secondary y-axis
        hydro_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create x-axis labels based on period
        if hydro_period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in hydro_filtered_df['Date']]
        elif hydro_period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in hydro_filtered_df['Date']]
        elif hydro_period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in hydro_filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in hydro_filtered_df['Date']]
        
        hydro_fig.add_trace(
            go.Bar(
                name="Hydro Power Volume",
                x=x_labels,
                y=hydro_filtered_df['Hydro'],
                marker_color='#08C179',
                hovertemplate=f"Period: %{{x}}<br>Hydro Volume: %{{y}} MWh<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add growth line
        hydro_fig.add_trace(
            go.Scatter(
                name=growth_title,
                x=x_labels,
                y=hydro_filtered_df['Growth'],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                hovertemplate=f"{growth_title}<br>Period: %{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>"
            ),
            secondary_y=True
        )
        
        hydro_fig.update_layout(
            title=f'{hydro_period} Hydro Power Volume with {growth_title}',
            hovermode='x unified',
            showlegend=True
        )
        
        hydro_fig.update_yaxes(title_text="Hydro Power Volume (MWh)", secondary_y=False)
        hydro_fig.update_yaxes(title_text=growth_title, secondary_y=True)
        hydro_fig.update_xaxes(title_text="Date")
        
        # Remove secondary y-axis gridlines
        hydro_fig = update_chart_layout_with_no_secondary_grid(hydro_fig)
        
        st.plotly_chart(hydro_fig, use_container_width=True)
        
        # Download data section
        st.subheader("📥 Download Data")
        hydro_download_df = hydro_filtered_df[['Date', 'Hydro', 'Growth']].copy()
        hydro_download_df['Period_Label'] = x_labels
        add_download_buttons(hydro_download_df, f"hydro_power_{hydro_period.lower()}_{hydro_growth_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}")

    # Stock Performance Chart for Hydro Sector
    st.subheader("📈 Hydro Sector Stocks Performance (YTD)")
    hydro_stocks = ['REE', 'PC1', 'HDG', 'VSH', 'TMP', 'CHP']
    hydro_stock_fig = create_stock_performance_chart(hydro_stocks, "Hydro Power")
    st.plotly_chart(hydro_stock_fig, use_container_width=True)


# Tab 5: Coal-fired Power
coal_tab_index = 4 if has_renewable_data else 3
with tabs[coal_tab_index]:
    # Calculate rating for coal power
    coal_power_df_rating = df[['Coals', 'Date']].copy()
    coal_power_df_rating = coal_power_df_rating.rename(columns={'Date': 'date'})
    rating, rating_details = calculate_power_rating(coal_power_df_rating, 'Coals', 'date')
    
    # Header with rating
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Coal-fired Power Analysis")
    with col2:
        if rating == "Positive":
            st.success(f"**Rating: {rating}**")
        elif rating == "Negative":
            st.error(f"**Rating: {rating}**")
        else:
            st.info(f"**Rating: {rating}**")
        st.caption(rating_details)
    
    # Controls
    coal_col1, coal_col2 = st.columns(2)
    
    with coal_col1:
        coal_chart_type = st.selectbox(
            "Select Chart Type:",
            ["Coal Costs", "Coal Power Volume"],
            key="coal_chart_type"
        )
    
    with coal_col2:
        coal_period = st.selectbox(
            "Select Time Period:",
            ["Monthly", "Quarterly", "Semi-annually", "Annually"],
            key="coal_period"
        )
    
    if coal_chart_type == "Coal Costs" and has_thermal_data:
        st.subheader("Coal Costs Analysis")
        
        # Filter out future dates beyond current date
        current_date = pd.Timestamp.now()
        available_years = sorted([year for year in thermal_df['Year'].unique()])
        
        # Year selector
        coal_show_years = st.multiselect(
            "Select years to display:",
            options=available_years,
            default=available_years,
            key="coal_show_years"
        )
        
        # Filter and aggregate coal cost data
        if coal_period == "Monthly":
            coal_filtered_df = thermal_df.groupby(['Year', 'Month'])[['coal cost for Vinh Tan (VND/ton)', 'coal cost for Mong Duong (VND/ton)']].mean().reset_index()
            coal_filtered_df['Date'] = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(coal_filtered_df['Year'], coal_filtered_df['Month'])])
        elif coal_period == "Quarterly":
            coal_filtered_df = thermal_df.groupby(['Year', 'Quarter'])[['coal cost for Vinh Tan (VND/ton)', 'coal cost for Mong Duong (VND/ton)']].mean().reset_index()
            coal_filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3:02d}-01" for y, q in zip(coal_filtered_df['Year'], coal_filtered_df['Quarter'])])
        elif coal_period == "Semi-annually":
            coal_filtered_df = thermal_df.groupby(['Year', 'Half'])[['coal cost for Vinh Tan (VND/ton)', 'coal cost for Mong Duong (VND/ton)']].mean().reset_index()
            coal_filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6:02d}-01" for y, h in zip(coal_filtered_df['Year'], coal_filtered_df['Half'])])
        else:  # Annually
            coal_filtered_df = thermal_df.groupby('Year')[['coal cost for Vinh Tan (VND/ton)', 'coal cost for Mong Duong (VND/ton)']].mean().reset_index()
            coal_filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in coal_filtered_df['Year']])
        
        # Filter years
        if coal_show_years:
            coal_filtered_df = coal_filtered_df[coal_filtered_df['Date'].dt.year.isin(coal_show_years)]
        
        # Create chart
        coal_fig = go.Figure()
        
        coal_types = ['coal cost for Vinh Tan (VND/ton)', 'coal cost for Mong Duong (VND/ton)']
        coal_names = ['Vinh Tan (Central)', 'Mong Duong (North)']
        colors = ['#08C179', '#97999B']
        
        if coal_period == "Annually":
            # Use grouped bar chart for annual data
            years = sorted([int(year) for year in coal_filtered_df['Date'].dt.year.unique()])
            for coal_idx, (coal_col, coal_name) in enumerate(zip(coal_types, coal_names)):
                coal_fig.add_trace(
                    go.Bar(
                        name=coal_name,
                        x=[str(year) for year in years],
                        y=[coal_filtered_df[coal_filtered_df['Date'].dt.year == year][coal_col].iloc[0] for year in years],
                        marker_color=colors[coal_idx],
                        hovertemplate=f"{coal_name}<br>Year: %{{x}}<br>Coal Cost: %{{y:,.0f}} VND/ton<extra></extra>"
                    )
                )
        else:
            # Use line chart for other periods
            for coal_idx, (coal_col, coal_name) in enumerate(zip(coal_types, coal_names)):
                coal_fig.add_trace(
                    go.Scatter(
                        name=coal_name,
                        x=coal_filtered_df['Date'],
                        y=coal_filtered_df[coal_col],
                        mode='lines+markers',
                        line=dict(color=colors[coal_idx], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"{coal_name}<br>Date: %{{x}}<br>Coal Cost: %{{y:,.0f}} VND/ton<extra></extra>"
                    )
                )
        
        coal_fig.update_layout(
            title=f"{coal_period} Coal Costs {'Analysis' if coal_period == 'Annually' else 'Over Time'}",
            xaxis_title="Year" if coal_period == "Annually" else "Date",
            yaxis_title="Coal Cost (VND/ton)",
            hovermode='closest' if coal_period == "Annually" else 'x unified',
            showlegend=True,
            barmode='group' if coal_period == "Annually" else None
        )
        
        st.plotly_chart(coal_fig, use_container_width=True)
        
        # Download data section for coal costs
        st.subheader("📥 Download Data")
        add_download_buttons(coal_filtered_df, f"coal_costs_{coal_period.lower()}")
        
    else:  # Coal Power Volume
        st.subheader("Coal Power Volume")
        
        # Growth type selector
        coal_growth_type = st.selectbox(
            "Select Growth Type:",
            ["Year-over-Year (YoY)", "Year-to-Date (YTD)"],
            key="coal_growth_type"
        )
        
        # Filter coal power data
        if coal_period == "Monthly":
            coal_filtered_df = df[['Date', 'Coals']].copy()
        elif coal_period == "Quarterly":
            coal_filtered_df = df.groupby(['Year', 'Quarter'])['Coals'].sum().reset_index()
            coal_filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3}-01" for y, q in zip(coal_filtered_df['Year'], coal_filtered_df['Quarter'])])
        elif coal_period == "Semi-annually":
            coal_filtered_df = df.groupby(['Year', 'Half'])['Coals'].sum().reset_index()
            coal_filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6}-01" for y, h in zip(coal_filtered_df['Year'], coal_filtered_df['Half'])])
        else:  # Annually
            coal_filtered_df = df.groupby('Year')['Coals'].sum().reset_index()
            coal_filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in coal_filtered_df['Year']])
        
        # Calculate growth with improved methods
        if coal_growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            coal_filtered_df['Growth'] = calculate_yoy_growth(coal_filtered_df, 'Coals', periods_map[coal_period])
            growth_title = "YoY Growth (%)"
        else:
            coal_filtered_df['Growth'] = calculate_ytd_growth(coal_filtered_df, 'Coals', 'Date', coal_period)
            growth_title = "YTD Growth (%)"
        
        # Create chart with secondary y-axis
        coal_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create x-axis labels based on period
        if coal_period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in coal_filtered_df['Date']]
        elif coal_period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in coal_filtered_df['Date']]
        elif coal_period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in coal_filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in coal_filtered_df['Date']]
        
        coal_fig.add_trace(
            go.Bar(
                name="Coal Power Volume",
                x=x_labels,
                y=coal_filtered_df['Coals'],
                marker_color='#08C179',
                hovertemplate=f"Period: %{{x}}<br>Coal Volume: %{{y}} MWh<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add growth line
        coal_fig.add_trace(
            go.Scatter(
                name=growth_title,
                x=x_labels,
                y=coal_filtered_df['Growth'],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                hovertemplate=f"{growth_title}<br>Period: %{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>"
            ),
            secondary_y=True
        )
        
        coal_fig.update_layout(
            title=f'{coal_period} Coal Power Volume with {growth_title}',
            hovermode='x unified',
            showlegend=True
        )
        
        coal_fig.update_yaxes(title_text="Coal Power Volume (MWh)", secondary_y=False)
        coal_fig.update_yaxes(title_text=growth_title, secondary_y=True)
        coal_fig.update_xaxes(title_text="Date")
        
        # Remove secondary y-axis gridlines
        coal_fig = update_chart_layout_with_no_secondary_grid(coal_fig)
        
        st.plotly_chart(coal_fig, use_container_width=True)
        
        # Download data section for coal volume
        st.subheader("📥 Download Data")
        coal_download_df = coal_filtered_df[['Date', 'Coals', 'Growth']].copy()
        coal_download_df['Period_Label'] = x_labels
        add_download_buttons(coal_download_df, f"coal_power_{coal_period.lower()}_{coal_growth_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}")

    # Stock Performance Chart for Coal Sector
    st.subheader("📈 Coal Sector Stocks Performance (YTD)")
    coal_stocks = ['POW', 'PPC']
    coal_stock_fig = create_stock_performance_chart(coal_stocks, "Coal Power")
    st.plotly_chart(coal_stock_fig, use_container_width=True)


# Tab 6: Gas-fired Power
gas_tab_index = 5 if has_renewable_data else 4
with tabs[gas_tab_index]:
    # Calculate rating for gas power
    gas_power_df_rating = df[['Gas', 'Date']].copy()
    gas_power_df_rating = gas_power_df_rating.rename(columns={'Date': 'date'})
    rating, rating_details = calculate_power_rating(gas_power_df_rating, 'Gas', 'date')
    
    # Header with rating
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("Gas-fired Power Analysis")
    with col2:
        if rating == "Positive":
            st.success(f"**Rating: {rating}**")
        elif rating == "Negative":
            st.error(f"**Rating: {rating}**")
        else:
            st.info(f"**Rating: {rating}**")
        st.caption(rating_details)
    
    # Controls
    gas_col1, gas_col2 = st.columns(2)
    
    with gas_col1:
        gas_chart_type = st.selectbox(
            "Select Chart Type:",
            ["Gas Costs", "Gas Power Volume"],
            key="gas_chart_type"
        )
    
    with gas_col2:
        gas_period = st.selectbox(
            "Select Time Period:",
            ["Monthly", "Quarterly", "Semi-annually", "Annually"],
            key="gas_period"
        )
    
    if gas_chart_type == "Gas Costs" and has_thermal_data:
        st.subheader("Gas Costs Analysis")
        
        # Filter out future dates beyond current date
        current_date = pd.Timestamp.now()
        available_years = sorted([year for year in thermal_df['Year'].unique()])
        
        # Year selector
        gas_show_years = st.multiselect(
            "Select years to display:",
            options=available_years,
            default=available_years,
            key="gas_show_years"
        )
        
        # Filter and aggregate gas cost data
        if gas_period == "Monthly":
            gas_filtered_df = thermal_df.groupby(['Year', 'Month'])[['Phu My Gas Cost (USD/MMBTU)', 'Nhon Trach Gas Cost (USD/MMBTU)']].mean().reset_index()
            gas_filtered_df['Date'] = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(gas_filtered_df['Year'], gas_filtered_df['Month'])])
        elif gas_period == "Quarterly":
            gas_filtered_df = thermal_df.groupby(['Year', 'Quarter'])[['Phu My Gas Cost (USD/MMBTU)', 'Nhon Trach Gas Cost (USD/MMBTU)']].mean().reset_index()
            gas_filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3:02d}-01" for y, q in zip(gas_filtered_df['Year'], gas_filtered_df['Quarter'])])
        elif gas_period == "Semi-annually":
            gas_filtered_df = thermal_df.groupby(['Year', 'Half'])[['Phu My Gas Cost (USD/MMBTU)', 'Nhon Trach Gas Cost (USD/MMBTU)']].mean().reset_index()
            gas_filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6:02d}-01" for y, h in zip(gas_filtered_df['Year'], gas_filtered_df['Half'])])
        else:  # Annually
            gas_filtered_df = thermal_df.groupby('Year')[['Phu My Gas Cost (USD/MMBTU)', 'Nhon Trach Gas Cost (USD/MMBTU)']].mean().reset_index()
            gas_filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in gas_filtered_df['Year']])
        
        # Filter years
        if gas_show_years:
            gas_filtered_df = gas_filtered_df[gas_filtered_df['Date'].dt.year.isin(gas_show_years)]
        
        # Create separate charts for each gas plant
        st.subheader("Phu My Gas Cost")
        phu_my_fig = go.Figure()
        
        gas_years = sorted([year for year in gas_filtered_df['Date'].dt.year.unique() if year in gas_show_years])
        colors = ['#0C4130', '#08C179', '#C0C1C2', '#97999B','#D3BB96', '#B78D51', '#014ABD']
        
        if gas_period == "Annually":
            # Use bar chart for annual data
            phu_my_fig.add_trace(
                go.Bar(
                    name="Phu My Gas Cost",
                    x=[str(int(year)) for year in gas_years],
                    y=[gas_filtered_df[gas_filtered_df['Date'].dt.year == year]['Phu My Gas Cost (USD/MMBTU)'].iloc[0] for year in gas_years],
                    marker_color='#08C179',
                    hovertemplate="Year: %{x}<br>Phu My Gas Cost: %{y:.2f} USD/MMBTU<extra></extra>"
                )
            )
        else:
            # Use line chart for other periods
            for i, year in enumerate(gas_years):
                year_data = gas_filtered_df[gas_filtered_df['Date'].dt.year == year].copy()
                
                if gas_period == "Monthly":
                    x_values = year_data['Date'].dt.strftime('%b')
                elif gas_period == "Quarterly":
                    x_values = 'Q' + year_data['Date'].dt.quarter.astype(str)
                elif gas_period == "Semi-annually":
                    x_values = 'H' + ((year_data['Date'].dt.month - 1) // 6 + 1).astype(str)
                
                phu_my_fig.add_trace(
                    go.Scatter(
                        name=f'{year}',
                        x=x_values,
                        y=year_data['Phu My Gas Cost (USD/MMBTU)'],
                        mode='lines+markers',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"Year: {year}<br>Period: %{{x}}<br>Phu My Gas Cost: %{{y:.2f}} USD/MMBTU<extra></extra>"
                    )
                )
        
        phu_my_fig.update_layout(
            title=f"{gas_period} Phu My Gas Cost {'Analysis' if gas_period == 'Annually' else 'Comparison by Year'}",
            xaxis_title="Year" if gas_period == "Annually" else f"{gas_period} Period",
            yaxis_title="Phu My Gas Cost (USD/MMBTU)",
            hovermode='closest' if gas_period == "Annually" else 'x unified',
            showlegend=False if gas_period == "Annually" else True
        )
        
        st.plotly_chart(phu_my_fig, use_container_width=True)
        
        # Nhon Trach Gas Cost Chart
        st.subheader("Nhon Trach Gas Cost")
        nhon_trach_fig = go.Figure()
        
        if gas_period == "Annually":
            # Use bar chart for annual data
            nhon_trach_fig.add_trace(
                go.Bar(
                    name="Nhon Trach Gas Cost",
                    x=[str(int(year)) for year in gas_years],
                    y=[gas_filtered_df[gas_filtered_df['Date'].dt.year == year]['Nhon Trach Gas Cost (USD/MMBTU)'].iloc[0] for year in gas_years],
                    marker_color='#08C179',
                    hovertemplate="Year: %{x}<br>Nhon Trach Gas Cost: %{y:.2f} USD/MMBTU<extra></extra>"
                )
            )
        else:
            # Use line chart for other periods
            for i, year in enumerate(gas_years):
                year_data = gas_filtered_df[gas_filtered_df['Date'].dt.year == year].copy()
                
                if gas_period == "Monthly":
                    x_values = year_data['Date'].dt.strftime('%b')
                elif gas_period == "Quarterly":
                    x_values = 'Q' + year_data['Date'].dt.quarter.astype(str)
                elif gas_period == "Semi-annually":
                    x_values = 'H' + ((year_data['Date'].dt.month - 1) // 6 + 1).astype(str)
                
                nhon_trach_fig.add_trace(
                    go.Scatter(
                        name=f'{year}',
                        x=x_values,
                        y=year_data['Nhon Trach Gas Cost (USD/MMBTU)'],
                        mode='lines+markers',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=4),
                        hovertemplate=f"Year: {year}<br>Period: %{{x}}<br>Nhon Trach Gas Cost: %{{y:.2f}} USD/MMBTU<extra></extra>"
                    )
                )
        
        nhon_trach_fig.update_layout(
            title=f"{gas_period} Nhon Trach Gas Cost {'Analysis' if gas_period == 'Annually' else 'Comparison by Year'}",
            xaxis_title="Year" if gas_period == "Annually" else f"{gas_period} Period",
            yaxis_title="Nhon Trach Gas Cost (USD/MMBTU)",
            hovermode='closest' if gas_period == "Annually" else 'x unified',
            showlegend=False if gas_period == "Annually" else True
        )
        
        st.plotly_chart(nhon_trach_fig, use_container_width=True)
        
        # Download data section for gas costs
        st.subheader("📥 Download Data")
        add_download_buttons(gas_filtered_df, f"gas_costs_{gas_period.lower()}")
        
    else:  # Gas Power Volume
        st.subheader("Gas Power Volume")
        
        # Growth type selector
        gas_growth_type = st.selectbox(
            "Select Growth Type:",
            ["Year-over-Year (YoY)", "Year-to-Date (YTD)"],
            key="gas_growth_type"
        )
        
        # Filter gas power data
        if gas_period == "Monthly":
            gas_filtered_df = df[['Date', 'Gas']].copy()
        elif gas_period == "Quarterly":
            gas_filtered_df = df.groupby(['Year', 'Quarter'])['Gas'].sum().reset_index()
            gas_filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3}-01" for y, q in zip(gas_filtered_df['Year'], gas_filtered_df['Quarter'])])
        elif gas_period == "Semi-annually":
            gas_filtered_df = df.groupby(['Year', 'Half'])['Gas'].sum().reset_index()
            gas_filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6}-01" for y, h in zip(gas_filtered_df['Year'], gas_filtered_df['Half'])])
        else:  # Annually
            gas_filtered_df = df.groupby('Year')['Gas'].sum().reset_index()
            gas_filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in gas_filtered_df['Year']])
        
        # Calculate growth with improved methods
        if gas_growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            gas_filtered_df['Growth'] = calculate_yoy_growth(gas_filtered_df, 'Gas', periods_map[gas_period])
            growth_title = "YoY Growth (%)"
        else:
            gas_filtered_df['Growth'] = calculate_ytd_growth(gas_filtered_df, 'Gas', 'Date', gas_period)
            growth_title = "YTD Growth (%)"
        
        # Create chart with secondary y-axis
        gas_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create x-axis labels based on period
        if gas_period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in gas_filtered_df['Date']]
        elif gas_period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in gas_filtered_df['Date']]
        elif gas_period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in gas_filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in gas_filtered_df['Date']]
        
        gas_fig.add_trace(
            go.Bar(
                name="Gas Power Volume",
                x=x_labels,
                y=gas_filtered_df['Gas'],
                marker_color='#08C179',
                hovertemplate=f"Period: %{{x}}<br>Gas Volume: %{{y}} MWh<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add growth line
        gas_fig.add_trace(
            go.Scatter(
                name=growth_title,
                x=x_labels,
                y=gas_filtered_df['Growth'],
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=4),
                hovertemplate=f"{growth_title}<br>Period: %{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>"
            ),
            secondary_y=True
        )
        
        gas_fig.update_layout(
            title=f'{gas_period} Gas Power Volume with {growth_title}',
            hovermode='x unified',
            showlegend=True
        )
        
        gas_fig.update_yaxes(title_text="Gas Power Volume (MWh)", secondary_y=False)
        gas_fig.update_yaxes(title_text=growth_title, secondary_y=True)
        gas_fig.update_xaxes(title_text="Date")
        
        # Remove secondary y-axis gridlines
        gas_fig = update_chart_layout_with_no_secondary_grid(gas_fig)
        
        st.plotly_chart(gas_fig, use_container_width=True)
        
        # Download data section for gas volume
        st.subheader("📥 Download Data")
        gas_download_df = gas_filtered_df[['Date', 'Gas', 'Growth']].copy()
        gas_download_df['Period_Label'] = x_labels
        add_download_buttons(gas_download_df, f"gas_power_{gas_period.lower()}_{gas_growth_type.lower().replace(' ', '_').replace('(', '').replace(')', '')}")

    # Stock Performance Chart for Gas Sector
    st.subheader("📈 Gas Sector Stocks Performance (YTD)")
    gas_stocks = ['POW', 'NT2', 'PGV', 'BTP']
    gas_stock_fig = create_stock_performance_chart(gas_stocks, "Gas Power")
    st.plotly_chart(gas_stock_fig, use_container_width=True)

