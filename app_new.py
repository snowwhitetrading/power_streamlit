import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import calendar
from plotly.subplots import make_subplots
import calendar

# Page configuration
st.set_page_config(page_title="Power Sector Dashboard", layout="wide")

# Title
st.title("Power Sector Dashboard")

# Load and process data
@st.cache_data
def load_data():
    # Load monthly power volume data
    df = pd.read_csv('Monthly_Power_Volume.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Half'] = (df['Date'].dt.month - 1) // 6 + 1
    
    # Load renewable data
    renewable_df = None
    has_renewable_data = False
    try:
        renewable_df = pd.read_csv('thong_so_vh.csv')
        renewable_df['date'] = pd.to_datetime(renewable_df['date'])
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
    
    with col2:
        growth_type = st.selectbox(
            "Select growth type:",
            ["Year-over-Year (YoY)", "Year-to-Date (YTD)"],
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
        filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in filtered_df['Year']])
    
    # Calculate total and growth for selected power types only
    filtered_df['Total'] = filtered_df[selected_power_types].sum(axis=1)
    
    if growth_type == "Year-over-Year (YoY)":
        periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
        filtered_df['Total_Growth'] = filtered_df['Total'].pct_change(periods=periods_map[period]) * 100
        growth_title = "YoY Growth (%)"
    else:
        filtered_df['Total_Growth'] = filtered_df['Total'].pct_change() * 100
        growth_title = "YTD Growth (%)"
    
    # Create chart with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add stacked bars for selected power types only
    power_types = ['Gas', 'Hydro', 'Coals', 'Renewables', 'Import & Diesel']
    power_names = ['Gas Power', 'Hydro Power', 'Coal Power', 'Renewables', 'Import & Diesel']
    colors = ['#FF6666', '#66B2FF', '#A9A9A9', '#66CC66', '#FFFF66']
    
    # Create x-axis labels based on period
    if period == "Monthly":
        x_labels = [d.strftime('%b %Y') for d in filtered_df['Date']]
    elif period == "Quarterly":
        x_labels = [f"Q{d.quarter} {d.year}" for d in filtered_df['Date']]
    elif period == "Semi-annually":
        x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in filtered_df['Date']]
    else:
        x_labels = [str(d.year) for d in filtered_df['Date']]
    
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
    
    st.plotly_chart(fig, use_container_width=True)

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
        
        # Create chart with separate lines for each year
        cgm_fig = go.Figure()
        
        years = sorted(cgm_filtered_df['Year'].unique())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
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
            title=f"{cgm_period} CGM Price Trend",
            xaxis_title="Time Period",
            yaxis_title="CGM Price (VND/kWh)",
            hovermode='x unified'
        )
        
        st.plotly_chart(cgm_fig, use_container_width=True)
    else:
        st.error("CGM price data not available.")

# Tab 3: Renewable Power (if available)
if has_renewable_data:
    with tabs[2]:
        st.header("Renewable Power Analysis")
        
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
            ren_growth_type = st.selectbox(
                "Select growth type:",
                ["Year-over-Year (YoY)", "Month-over-Month (MoM)"],
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
            ren_filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in ren_filtered_df['Year']])
        
        # Calculate total renewable volume for selected types only
        ren_filtered_df['Total_Renewable'] = ren_filtered_df[selected_renewable_cols].sum(axis=1)
        
        # Calculate growth
        if ren_growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            ren_filtered_df['Total_Growth'] = ren_filtered_df['Total_Renewable'].pct_change(periods=periods_map[ren_period]) * 100
            growth_title = "YoY Growth (%)"
        else:  # Month-over-Month (MoM)
            ren_filtered_df['Total_Growth'] = ren_filtered_df['Total_Renewable'].pct_change() * 100
            growth_title = "MoM Growth (%)"
        
        # Create chart with secondary y-axis
        ren_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add stacked bars for selected types only
        ren_names_map = {
            "dien_gio_mkWh": "Wind Power",
            "dmt_trang_trai_mkWh": "Solar Farm", 
            "dmt_mai_thuong_pham_mkWh": "Rooftop Solar"
        }
        ren_colors = ['#A9A9A9', '#FFFF66', '#66CC66']
        
        # Create x-axis labels based on period
        if ren_period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in ren_filtered_df['Date']]
        elif ren_period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in ren_filtered_df['Date']]
        elif ren_period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in ren_filtered_df['Date']]
        else:
            x_labels = [str(d.year) for d in ren_filtered_df['Date']]
        
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
        
        st.plotly_chart(ren_fig, use_container_width=True)

# Tab 4: Hydro Power
hydro_tab_index = 3 if has_renewable_data else 2
with tabs[hydro_tab_index]:
    st.header("Hydro Power Analysis")
    
    # Controls
    hydro_col1, hydro_col2 = st.columns(2)
    
    with hydro_col1:
        hydro_chart_type = st.selectbox(
            "Select Chart Type:",
            ["Total Capacity Comparison", "Hydro Power Volume"],
            key="hydro_chart_type"
        )
    
    with hydro_col2:
        if hydro_chart_type == "Total Capacity Comparison" and has_reservoir_data:
            # Create region translation dictionary
            region_translation = {
                "Đông Bắc Bộ": "Northeast",
                "Tây Bắc Bộ": "Northwest", 
                "Bắc Trung Bộ": "North Central",
                "Nam Trung Bộ": "South Central",
                "Tây Nguyên": "Central Highlands"
            }
            
            # Add English region names to reservoir data if not already done
            if 'region_en' not in reservoir_df.columns:
                reservoir_df['region_en'] = reservoir_df['region'].map(region_translation)
            
            hydro_region = st.selectbox(
                "Select Region:",
                sorted(reservoir_df['region_en'].dropna().unique()),
                key="hydro_region"
            )
        else:
            hydro_period = st.selectbox(
                "Select Time Period:",
                ["Monthly", "Quarterly", "Semi-annually", "Annually"],
                key="hydro_period"
            )
    
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
        
        # Create comparison chart
        capacity_fig = go.Figure()
        
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
                        marker_color='#66B2FF' if year == 2024 else '#A9A9A9',
                        hovertemplate=f"{year}<br>Month: %{{x}}<br>Avg Capacity: %{{y:.1f}} Million m³<extra></extra>"
                    )
                )
        
        capacity_fig.update_layout(
            title=f"Monthly Total Capacity Comparison (2024 vs 2025) - {hydro_region}",
            xaxis_title="Month",
            yaxis_title="Average Total Capacity (Million m³)",
            hovermode='x unified',
            showlegend=True,
            barmode='group'
        )
        
        st.plotly_chart(capacity_fig, use_container_width=True)
        
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
            hydro_filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in hydro_filtered_df['Year']])
        
        # Calculate growth
        if hydro_growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            hydro_filtered_df['Growth'] = hydro_filtered_df['Hydro'].pct_change(periods=periods_map[hydro_period]) * 100
            growth_title = "YoY Growth (%)"
        else:
            hydro_filtered_df['Growth'] = hydro_filtered_df['Hydro'].pct_change() * 100
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
            x_labels = [str(d.year) for d in hydro_filtered_df['Date']]
        
        hydro_fig.add_trace(
            go.Bar(
                name="Hydro Power Volume",
                x=x_labels,
                y=hydro_filtered_df['Hydro'],
                marker_color='#66B2FF',
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
        
        st.plotly_chart(hydro_fig, use_container_width=True)

# Tab 5: Coal-fired Power
coal_tab_index = 4 if has_renewable_data else 3
with tabs[coal_tab_index]:
    st.header("Coal-fired Power Analysis")
    
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
            coal_filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in coal_filtered_df['Year']])
        
        # Filter years
        if coal_show_years:
            coal_filtered_df = coal_filtered_df[coal_filtered_df['Date'].dt.year.isin(coal_show_years)]
        
        # Create chart
        coal_fig = go.Figure()
        
        coal_types = ['coal cost for Vinh Tan (VND/ton)', 'coal cost for Mong Duong (VND/ton)']
        coal_names = ['Vinh Tan (Central)', 'Mong Duong (North)']
        colors = ['#08C179', '#97999B']
        
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
            title=f"{coal_period} Coal Costs Over Time",
            xaxis_title="Date",
            yaxis_title="Coal Cost (VND/ton)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(coal_fig, use_container_width=True)
        
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
            coal_filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in coal_filtered_df['Year']])
        
        # Calculate growth
        if coal_growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            coal_filtered_df['Growth'] = coal_filtered_df['Coals'].pct_change(periods=periods_map[coal_period]) * 100
            growth_title = "YoY Growth (%)"
        else:
            coal_filtered_df['Growth'] = coal_filtered_df['Coals'].pct_change() * 100
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
            x_labels = [str(d.year) for d in coal_filtered_df['Date']]
        
        coal_fig.add_trace(
            go.Bar(
                name="Coal Power Volume",
                x=x_labels,
                y=coal_filtered_df['Coals'],
                marker_color='#A9A9A9',
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
        
        st.plotly_chart(coal_fig, use_container_width=True)

# Tab 6: Gas-fired Power
gas_tab_index = 5 if has_renewable_data else 4
with tabs[gas_tab_index]:
    st.header("Gas-fired Power Analysis")
    
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
            gas_filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in gas_filtered_df['Year']])
        
        # Filter years
        if gas_show_years:
            gas_filtered_df = gas_filtered_df[gas_filtered_df['Date'].dt.year.isin(gas_show_years)]
        
        # Create separate charts for each gas plant
        st.subheader("Phu My Gas Cost")
        phu_my_fig = go.Figure()
        
        gas_years = sorted([year for year in gas_filtered_df['Date'].dt.year.unique() if year in gas_show_years])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, year in enumerate(gas_years):
            year_data = gas_filtered_df[gas_filtered_df['Date'].dt.year == year].copy()
            
            if gas_period == "Monthly":
                x_values = year_data['Date'].dt.strftime('%b')
            elif gas_period == "Quarterly":
                x_values = 'Q' + year_data['Date'].dt.quarter.astype(str)
            elif gas_period == "Semi-annually":
                x_values = 'H' + ((year_data['Date'].dt.month - 1) // 6 + 1).astype(str)
            else:  # Annually
                x_values = year_data['Date'].dt.year
            
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
            title=f"{gas_period} Phu My Gas Cost Comparison by Year",
            xaxis_title=f"{gas_period} Period",
            yaxis_title="Phu My Gas Cost (USD/MMBTU)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(phu_my_fig, use_container_width=True)
        
        # Nhon Trach Gas Cost Chart
        st.subheader("Nhon Trach Gas Cost")
        nhon_trach_fig = go.Figure()
        
        for i, year in enumerate(gas_years):
            year_data = gas_filtered_df[gas_filtered_df['Date'].dt.year == year].copy()
            
            if gas_period == "Monthly":
                x_values = year_data['Date'].dt.strftime('%b')
            elif gas_period == "Quarterly":
                x_values = 'Q' + year_data['Date'].dt.quarter.astype(str)
            elif gas_period == "Semi-annually":
                x_values = 'H' + ((year_data['Date'].dt.month - 1) // 6 + 1).astype(str)
            else:  # Annually
                x_values = year_data['Date'].dt.year
            
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
            title=f"{gas_period} Nhon Trach Gas Cost Comparison by Year",
            xaxis_title=f"{gas_period} Period",
            yaxis_title="Nhon Trach Gas Cost (USD/MMBTU)",
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(nhon_trach_fig, use_container_width=True)
        
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
            gas_filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in gas_filtered_df['Year']])
        
        # Calculate growth
        if gas_growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            gas_filtered_df['Growth'] = gas_filtered_df['Gas'].pct_change(periods=periods_map[gas_period]) * 100
            growth_title = "YoY Growth (%)"
        else:
            gas_filtered_df['Growth'] = gas_filtered_df['Gas'].pct_change() * 100
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
            x_labels = [str(d.year) for d in gas_filtered_df['Date']]
        
        gas_fig.add_trace(
            go.Bar(
                name="Gas Power Volume",
                x=x_labels,
                y=gas_filtered_df['Gas'],
                marker_color='#FF6666',
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
        
        st.plotly_chart(gas_fig, use_container_width=True)
