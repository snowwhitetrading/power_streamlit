import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Load the data
df = pd.read_csv('Monthly_Power_Volume.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Title
st.title("Power Volume Chart")

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

# Print data to understand calculations
st.write("### Data Analysis")
with st.expander(f"View detailed {growth_type} calculation"):
    st.write(f"Power values and {growth_title}:")
    analysis_df = filtered_df[['Date', 'Total_Power', 'Growth']].copy()
    analysis_df['Year'] = analysis_df['Date'].dt.year
    if period == "Monthly":
        analysis_df['Period'] = analysis_df['Date'].dt.strftime('%B')
    elif period == "Quarterly":
        analysis_df['Period'] = 'Q' + analysis_df['Date'].dt.quarter.astype(str)
    elif period == "Semi-annually":
        analysis_df['Period'] = 'H' + ((analysis_df['Date'].dt.month - 1) // 6 + 1).astype(str)
    else:
        analysis_df['Period'] = 'Year'
    
    st.dataframe(analysis_df[['Date', 'Period', 'Total_Power', 'Growth']].round(2))

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

# Display the chart
st.plotly_chart(fig, use_container_width=True)

# Add explanatory notes
if period != "Monthly":
    period_examples = {
        "Quarterly": "Q1 2024 vs Q1 2023",
        "Semi-annually": "H1 2024 vs H1 2023",
        "Annually": "2024 vs 2023"
    }
# st.info(f"""Data is aggregated {period.lower()}. 
#              YoY growth compares total power in each period with the same period from previous year.
#              For example, {period_examples[period]} for {period.lower()} data.""")

# st.write("### Note about 2023 Data")
# st.warning("""The YoY growth shows 0% for 2023 because the data for 2023 is exactly the same as 2022. 
#            Looking at the source data, we can see that the values for each month in 2023 are identical to the corresponding months in 2022, 
#            which results in zero year-over-year growth.""")