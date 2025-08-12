import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Load the data
df = pd.read_csv('Monthly_Power_Volume.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Load and process weighted average price data
price_df = pd.read_csv('weighted_average_prices.csv')
price_df['date'] = pd.to_datetime(price_df['date'])

# Prepare price data aggregation
price_df['Year'] = price_df['date'].dt.year
price_df['Month'] = price_df['date'].dt.month
price_df['Quarter'] = price_df['date'].dt.quarter
price_df['Half'] = (price_df['date'].dt.month - 1) // 6 + 1

# Load water reservoir data
try:
    reservoir_df = pd.read_csv('water_reservoir.csv')
    reservoir_df['date_time'] = pd.to_datetime(reservoir_df['date_time'], dayfirst=True)
    reservoir_df['total_capacity'] = pd.to_numeric(reservoir_df['total_capacity'], errors='coerce')
    has_reservoir_data = True
    print(f"Successfully loaded water reservoir data with {len(reservoir_df)} rows")
except Exception as e:
    has_reservoir_data = False
    print(f"Error loading water reservoir data: {e}")
    st.error(f"Could not load water reservoir data: {e}")

# Title
st.title("Power Sector Dashboard")

# Navigation tabs
if has_reservoir_data:
    tab1, tab2, tab3 = st.tabs(["📊 Power Volume", "💰 CGM Price", "🏭 Water Reservoirs"])
else:
    tab1, tab2 = st.tabs(["📊 Power Volume", "💰 CGM Price"])

with tab1:
    st.header("Power Volume Analysis")
    
    # Create columns for controls
    col1, col2, col3 = st.columns(3)

    # Select columns to display
    with col1:
        energy_types = df.columns[1:]  # Exclude 'Date'
        selected_types = st.multiselect("Select energy types to display:", energy_types, default=list(energy_types))

    # Select time period
    with col2:
        period = st.selectbox(
            "Select time period:",
            ["Monthly", "Quarterly", "Semi-annually", "Annually"],
            index=0
        )

    # Select growth type
    with col3:
        growth_type = st.selectbox(
            "Select growth type:",
            ["Year-over-Year (YoY)", "Year-to-Date (YTD)"],
            index=0
        )

    # Prepare data aggregation
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Half'] = (df['Date'].dt.month - 1) // 6 + 1

    # Filter and aggregate data based on selected period
    if period == "Monthly":
        filtered_df = df[['Date'] + selected_types]
        date_col = 'Date'
    elif period == "Quarterly":
        filtered_df = df.groupby(['Year', 'Quarter'])[selected_types].sum().reset_index()
        filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3}-01" for y, q in zip(filtered_df['Year'], filtered_df['Quarter'])])
        date_col = 'Date'
    elif period == "Semi-annually":
        filtered_df = df.groupby(['Year', 'Half'])[selected_types].sum().reset_index()
        filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])
        date_col = 'Date'
    else:  # Annually
        filtered_df = df.groupby('Year')[selected_types].sum().reset_index()
        filtered_df['Date'] = pd.to_datetime(filtered_df['Year'].astype(str) + '-01-01')
        date_col = 'Date'

    # Calculate total power
    filtered_df['Total_Power'] = filtered_df[selected_types].sum(axis=1)

    # Calculate growth based on selected type
    if growth_type == "Year-over-Year (YoY)":
        # YoY growth based on the selected period
        periods_map = {
            "Monthly": 12,
            "Quarterly": 4,
            "Semi-annually": 2,
            "Annually": 1
        }
        filtered_df['Growth'] = filtered_df['Total_Power'].pct_change(periods=periods_map[period]) * 100
        growth_title = "YoY Growth (%)"
    else:  # Year-to-Date (YTD)
        # Calculate YTD growth
        filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year
        filtered_df['Period_Num'] = pd.to_datetime(filtered_df['Date']).dt.month
        if period == "Quarterly":
            filtered_df['Period_Num'] = pd.to_datetime(filtered_df['Date']).dt.quarter
        elif period == "Semi-annually":
            filtered_df['Period_Num'] = (pd.to_datetime(filtered_df['Date']).dt.month - 1) // 6 + 1
        
        # Calculate YTD sums for each year
        ytd_sums = filtered_df.groupby(['Year', 'Period_Num'])['Total_Power'].sum().groupby('Year').cumsum().reset_index()
        ytd_sums_prev_year = ytd_sums.copy()
        ytd_sums_prev_year['Year'] += 1
        
        # Merge current and previous year YTD sums
        merged_ytd = pd.merge(ytd_sums, ytd_sums_prev_year, 
                             on=['Year', 'Period_Num'], 
                             how='left', 
                             suffixes=('', '_prev'))
        
        # Calculate YTD growth
        merged_ytd['YTD_Growth'] = ((merged_ytd['Total_Power'] - merged_ytd['Total_Power_prev']) / 
                                   merged_ytd['Total_Power_prev'] * 100)
        
        # Merge growth back to filtered_df
        filtered_df = pd.merge(filtered_df, 
                              merged_ytd[['Year', 'Period_Num', 'YTD_Growth']], 
                              on=['Year', 'Period_Num'], 
                              how='left')
        filtered_df['Growth'] = filtered_df['YTD_Growth']
        growth_title = "YTD Growth (%)"

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Define color mapping for energy types
    color_map = {
        'Gas': '#ff6666',  # Medium red
        'Coals': '#999999',  # Medium grey
        'Renewables': '#66cc66',  # Medium green
        'Import & Diesel': '#ffd966',  # Medium yellow
        'Hydro': '#6699cc'  # Medium blue
    }

    # Add stacked bars
    for energy_type in selected_types:
        fig.add_trace(
            go.Bar(
                name=energy_type,
                x=filtered_df['Date'],
                y=filtered_df[energy_type],
                hovertemplate="%{y:.0f}",
                marker_color=color_map.get(energy_type, 'blue')  # Default to blue if type not in map
            ),
            secondary_y=False
        )

    # Add growth line
    fig.add_trace(
        go.Scatter(
            name=growth_title,
            x=filtered_df['Date'],
            y=filtered_df['Growth'],
            line=dict(color='red', width=2),
            hovertemplate="%{y:.1f}%",
        ),
        secondary_y=True
    )

    # Update layout
    fig.update_layout(
        title=f'{period} Power Volume by Type with {growth_title}',
        barmode='stack',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        showlegend=True
    )

    # Update axes
    fig.update_yaxes(
        title_text="Power Volume", 
        secondary_y=False,
        showgrid=False,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        rangemode='nonnegative',  # Prevents space below 0
        range=[0, None]  # Start from 0
    )
    fig.update_yaxes(
        title_text=growth_title, 
        secondary_y=True,
        showgrid=False,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    fig.update_xaxes(
        title_text="Date",
        showgrid=False,
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black'
    )

    # Display the power volume chart
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("CGM Price Analysis")
    
    # Create controls for price chart
    price_col1, price_col2 = st.columns(2)

    # Select time period for price chart
    with price_col1:
        price_period = st.selectbox(
            "Select time period for price:",
            ["Monthly", "Quarterly", "Semi-annually", "Annually"],  # Removed "Weekly"
            index=0,  # Default to Monthly
            key="price_period"
        )

    # Add option to show/hide individual years
    with price_col2:
        show_years = st.multiselect(
            "Select years to display:",
            options=sorted(price_df['Year'].unique()),
            default=sorted(price_df['Year'].unique()),
            key="show_years"
        )

    # Filter and aggregate price data based on selected period
    if price_period == "Monthly":
        price_filtered_df = price_df.groupby(['Year', 'Month'])['weighted_avg_price'].mean().reset_index()
        price_filtered_df['Date'] = pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(price_filtered_df['Year'], price_filtered_df['Month'])])
    elif price_period == "Quarterly":
        price_filtered_df = price_df.groupby(['Year', 'Quarter'])['weighted_avg_price'].mean().reset_index()
        price_filtered_df['Date'] = pd.to_datetime([f"{y}-{q*3:02d}-01" for y, q in zip(price_filtered_df['Year'], price_filtered_df['Quarter'])])
    elif price_period == "Semi-annually":
        price_filtered_df = price_df.groupby(['Year', 'Half'])['weighted_avg_price'].mean().reset_index()
        price_filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6:02d}-01" for y, h in zip(price_filtered_df['Year'], price_filtered_df['Half'])])
    else:  # Annually
        price_filtered_df = price_df.groupby('Year')['weighted_avg_price'].mean().reset_index()
        price_filtered_df['Date'] = pd.to_datetime([f"{y}-01-01" for y in price_filtered_df['Year']])

    # Filter years based on user selection
    if show_years:
        price_filtered_df = price_filtered_df[price_filtered_df['Date'].dt.year.isin(show_years)]

    # Create price chart
    price_fig = go.Figure()

    # Add separate line for each year
    years = sorted([year for year in price_filtered_df['Date'].dt.year.unique() if year in show_years])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for i, year in enumerate(years):
        year_data = price_filtered_df[price_filtered_df['Date'].dt.year == year].copy()
        
        # Create proper x-axis values based on period
        if price_period == "Monthly":
            year_data['Month_Name'] = year_data['Date'].dt.strftime('%b')  # Jan, Feb, Mar, etc.
            x_values = year_data['Month_Name']
        elif price_period == "Quarterly":
            year_data['Quarter_Name'] = 'Q' + year_data['Date'].dt.quarter.astype(str)
            x_values = year_data['Quarter_Name']
        elif price_period == "Semi-annually":
            year_data['Half_Name'] = 'H' + ((year_data['Date'].dt.month - 1) // 6 + 1).astype(str)
            x_values = year_data['Half_Name']
        else:  # Annually
            x_values = year_data['Date'].dt.year
        
        price_fig.add_trace(
            go.Scatter(
                name=f'{year}',
                x=x_values,
                y=year_data['weighted_avg_price'],
                mode='lines+markers',
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4),
                hovertemplate=f"Year: {year}<br>Period: %{{x}}<br>Price: %{{y:.2f}}<extra></extra>",
            ),
        )

    # Update price chart layout
    price_fig.update_layout(
        title=f"{price_period} Average Price Comparison by Year",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        ),
        xaxis=dict(
            title=f"{price_period.rstrip('ly')} Period",
            showgrid=True,
            gridcolor='lightgrey',
            ticks="outside",
            ticklen=5,
            tickwidth=1,
            showline=True,
            linewidth=1,
            linecolor='black'
        )
    )

    # Update price axes
    price_fig.update_yaxes(
        title_text="Price",
        showgrid=True,
        gridcolor='lightgrey',
        ticks="outside",
        ticklen=5,
        tickwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black'
    )

    # Display the price chart
    st.plotly_chart(price_fig, use_container_width=True)

# Water Reservoir Analysis Tab
if has_reservoir_data:
    with tab3:
        st.header("Water Reservoir Capacity Analysis")
        
        # Prepare reservoir data
        reservoir_df['Year'] = reservoir_df['date_time'].dt.year
        reservoir_df['Month'] = reservoir_df['date_time'].dt.month
        reservoir_df['Quarter'] = reservoir_df['date_time'].dt.quarter
        reservoir_df['Half'] = (reservoir_df['date_time'].dt.month - 1) // 6 + 1
        reservoir_df['Date'] = reservoir_df['date_time'].dt.date
        
        # Filter for 2024 and 2025 only
        reservoir_filtered = reservoir_df[reservoir_df['Year'].isin([2024, 2025])]
        
        if len(reservoir_filtered) > 0:
            # 1. Daily aggregation by region
            daily_capacity = reservoir_filtered.groupby(['Date', 'region'])['total_capacity'].sum().reset_index()
            daily_capacity['Date'] = pd.to_datetime(daily_capacity['Date'])
            daily_capacity['Year'] = daily_capacity['Date'].dt.year
            daily_capacity['Month'] = daily_capacity['Date'].dt.month
            daily_capacity['Quarter'] = daily_capacity['Date'].dt.quarter
            daily_capacity['Half'] = (daily_capacity['Date'].dt.month - 1) // 6 + 1
            
            # 2. Calculate averages for different periods
            def calculate_period_averages(df, period_type):
                if period_type == "Monthly":
                    return df.groupby(['Year', 'Month', 'region'])['total_capacity'].mean().reset_index()
                elif period_type == "Quarterly":
                    return df.groupby(['Year', 'Quarter', 'region'])['total_capacity'].mean().reset_index()
                elif period_type == "Semi-annually":
                    return df.groupby(['Year', 'Half', 'region'])['total_capacity'].mean().reset_index()
            
            # Controls for reservoir analysis
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                available_regions = sorted(reservoir_filtered['region'].unique())
                selected_region = st.selectbox(
                    "Select Region:",
                    options=available_regions,
                    key="reservoir_region"
                )
            
            with res_col2:
                time_aggregation = st.selectbox(
                    "Select Time Period:",
                    options=["Monthly", "Quarterly", "Semi-annually"],
                    key="reservoir_period"
                )
            
            # Calculate averages for selected period
            period_averages = calculate_period_averages(daily_capacity, time_aggregation)
            
            # Filter for selected region
            region_data = period_averages[period_averages['region'] == selected_region]
            
            if len(region_data) > 0:
                # Create comparison chart (2024 vs 2025)
                reservoir_fig = go.Figure()
                
                # Separate data by year
                data_2024 = region_data[region_data['Year'] == 2024]
                data_2025 = region_data[region_data['Year'] == 2025]
                
                # Create x-axis labels based on period
                if time_aggregation == "Monthly":
                    x_col = 'Month'
                    x_labels_2024 = [f"Month {m}" for m in data_2024['Month']]
                    x_labels_2025 = [f"Month {m}" for m in data_2025['Month']]
                elif time_aggregation == "Quarterly":
                    x_col = 'Quarter'
                    x_labels_2024 = [f"Q{q}" for q in data_2024['Quarter']]
                    x_labels_2025 = [f"Q{q}" for q in data_2025['Quarter']]
                else:  # Semi-annually
                    x_col = 'Half'
                    x_labels_2024 = [f"H{h}" for h in data_2024['Half']]
                    x_labels_2025 = [f"H{h}" for h in data_2025['Half']]
                
                # Add 2024 data
                if len(data_2024) > 0:
                    reservoir_fig.add_trace(
                        go.Bar(
                            name='2024',
                            x=x_labels_2024,
                            y=data_2024['total_capacity'],
                            marker_color='#1f77b4',
                            hovertemplate="Year: 2024<br>Period: %{x}<br>Avg Capacity: %{y:.2f}<extra></extra>"
                        )
                    )
                
                # Add 2025 data
                if len(data_2025) > 0:
                    reservoir_fig.add_trace(
                        go.Bar(
                            name='2025',
                            x=x_labels_2025,
                            y=data_2025['total_capacity'],
                            marker_color='#ff7f0e',
                            hovertemplate="Year: 2025<br>Period: %{x}<br>Avg Capacity: %{y:.2f}<extra></extra>"
                        )
                    )
                
                # Update layout
                reservoir_fig.update_layout(
                    title=f"{time_aggregation} Average Total Capacity - {selected_region} (2024 vs 2025)",
                    xaxis_title=f"{time_aggregation} Period",
                    yaxis_title="Average Total Capacity",
                    barmode='group',
                    hovermode='x unified',
                    showlegend=True
                )
                
                # Display the chart
                st.plotly_chart(reservoir_fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if len(data_2024) > 0:
                        avg_2024 = data_2024['total_capacity'].mean()
                        st.metric(label="2024 Average Capacity", value=f"{avg_2024:.2f}")
                
                with col2:
                    if len(data_2025) > 0:
                        avg_2025 = data_2025['total_capacity'].mean()
                        st.metric(label="2025 Average Capacity", value=f"{avg_2025:.2f}")
                
                with col3:
                    if len(data_2024) > 0 and len(data_2025) > 0:
                        change = ((avg_2025 - avg_2024) / avg_2024) * 100
                        st.metric(label="Year-over-Year Change", value=f"{change:.1f}%")
                
                # Show detailed data table
                if st.expander("View Detailed Data"):
                    st.subheader("Monthly/Quarterly/Semi-annual Averages")
                    display_data = region_data.pivot(index=x_col, columns='Year', values='total_capacity')
                    st.dataframe(display_data)
            
            else:
                st.warning(f"No data available for region: {selected_region}")
        else:
            st.warning("No reservoir data available for 2024-2025 period.")

# Add explanatory notes
if period != "Monthly":
    period_examples = {
        "Quarterly": "Q1 2024 vs Q1 2023",
        "Semi-annually": "H1 2024 vs H1 2023",
        "Annually": "2024 vs 2023"
    }
