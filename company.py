import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import calendar
import numpy as np
from plotly.subplots import make_subplots
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Helper functions for downloads
@st.cache_data
def convert_df_to_excel(df, sheet_name="Data"):
    """Convert dataframe to Excel bytes for download"""
    import io
    from datetime import datetime
    
    output = io.BytesIO()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Add some cell formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Adjust column widths
        for idx, col in enumerate(df.columns):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(str(col))
            ) + 2
            worksheet.set_column(idx, idx, min(max_len, 50))
    
    return output.getvalue()

@st.cache_data  
def convert_df_to_csv(df):
    """Convert dataframe to CSV string for download"""
    return df.to_csv(index=False).encode('utf-8')

def add_download_buttons(df, filename_prefix, container=None):
    """Add download buttons for Excel and CSV"""
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    display_container = container if container else st
    
    col1, col2 = display_container.columns(2)
    
    with col1:
        if display_container.download_button(
            label="📊 Download as Excel",
            data=convert_df_to_excel(df),
            file_name=f"{filename_prefix}_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"excel_{filename_prefix}_{timestamp}"
        ):
            display_container.success("Excel file downloaded successfully!")
    
    with col2:
        if display_container.download_button(
            label="📄 Download as CSV",
            data=convert_df_to_csv(df),
            file_name=f"{filename_prefix}_{timestamp}.csv",
            mime="text/csv",
            key=f"csv_{filename_prefix}_{timestamp}"
        ):
            display_container.success("CSV file downloaded successfully!")

def update_chart_layout_with_no_secondary_grid(fig):
    """Remove gridlines from secondary y-axis while keeping the axis"""
    fig.update_layout(
        yaxis2=dict(showgrid=False)
    )
    return fig

def calculate_ytd_growth(df, value_col, date_col, period_type):
    """Calculate proper YTD growth from cumulative values from beginning of year"""
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['Year'] = df[date_col].dt.year
        
        if period_type == "Monthly":
            df['Month'] = df[date_col].dt.month
            df = df.sort_values([date_col])
            
            # Calculate cumulative within each year
            df['YTD_Value'] = df.groupby('Year')[value_col].cumsum()
            
            # Calculate YTD growth
            ytd_growth = []
            for _, row in df.iterrows():
                year = row['Year']
                month = row['Month']
                current_ytd = row['YTD_Value']
                
                # Find same period last year
                last_year_data = df[(df['Year'] == year - 1) & (df['Month'] == month)]
                
                if not last_year_data.empty and current_ytd != 0:
                    last_year_ytd = last_year_data['YTD_Value'].iloc[0]
                    if last_year_ytd != 0:
                        growth = ((current_ytd - last_year_ytd) / last_year_ytd) * 100
                        ytd_growth.append(growth)
                    else:
                        ytd_growth.append(0)
                else:
                    ytd_growth.append(0)
            
            return ytd_growth
        else:
            # For other periods, calculate simple growth
            df['Growth'] = df[value_col].pct_change() * 100
            return df['Growth'].fillna(0).tolist()
            
    except Exception as e:
        st.error(f"Error calculating YTD growth: {e}")
        return [0] * len(df)

def calculate_yoy_growth(df, value_col, periods):
    """Calculate YoY growth only when sufficient historical data exists"""
    try:
        df = df.copy()
        
        # Ensure the column is numeric
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
        
        growth_values = []
        for i in range(len(df)):
            if i >= periods:  # Ensure we have enough historical data
                current_value = df.iloc[i][value_col]
                previous_value = df.iloc[i - periods][value_col]
                
                if previous_value != 0:
                    growth = ((current_value - previous_value) / previous_value) * 100
                    growth_values.append(growth)
                else:
                    growth_values.append(0)
            else:
                growth_values.append(0)  # No growth for insufficient data
        
        return growth_values
    except Exception as e:
        st.error(f"Error calculating YoY growth: {e}")
        return [0] * len(df)

# Data loading functions
@st.cache_data
def load_company_data(company_name):
    """Load company data from CSV files"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if company_name.upper() == "POW":
            # POW data from volume_pow_monthly.csv
            file_path = os.path.join(script_dir, 'data', 'volume_pow_monthly.csv')
            df = pd.read_csv(file_path)
            
            # Process POW data
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Quarter'] = df['Date'].dt.quarter
            df['Half'] = (df['Date'].dt.month - 1) // 6 + 1
            
            return df, True
        else:
            # Other companies from company_xxx_monthly.csv
            file_path = os.path.join(script_dir, 'data', f'company_{company_name.lower()}_monthly.csv')
            
            if not os.path.exists(file_path):
                st.warning(f"Data file for {company_name} not found: {file_path}")
                return None, False
            
            df = pd.read_csv(file_path)
            
            # Filter out rows with no data (all numeric columns are NaN)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Keep rows where at least one numeric column has data
                df = df.dropna(subset=numeric_cols, how='all')
            
            # Process quarterly data from Date column
            if 'Date' in df.columns:
                # The Date column contains period info like "1Q2021"
                periods = df['Date'].astype(str)
                dates = []
                quarters = []
                years = []
                halves = []
                
                for period in periods:
                    if 'Q' in period:
                        quarter = int(period[0])
                        year = int(period[2:])
                        # Convert quarter to approximate date (middle of quarter)
                        month = quarter * 3 - 1  # Q1->2, Q2->5, Q3->8, Q4->11
                        dates.append(pd.to_datetime(f"{year}-{month:02d}-15"))
                        quarters.append(quarter)
                        years.append(year)
                        # Calculate half year: Q1,Q2 = H1, Q3,Q4 = H2
                        halves.append(1 if quarter <= 2 else 2)
                
                df['Date'] = dates
                df['Year'] = years
                df['Month'] = [d.month for d in dates]
                df['Quarter'] = quarters
                df['Half'] = halves
            
            return df, True
            
    except Exception as e:
        st.error(f"Error loading {company_name} data: {e}")
        return None, False

def get_company_columns(company_name, df):
    """Get relevant columns for each company based on their data structure"""
    if company_name.upper() == "POW":
        # POW specific columns
        power_plants = ['Ca Mau', 'Nhon Trach 1', 'Nhon Trach 2 (NT2)', 'Hua Na', 'Dak Drinh', 'Vung Ang']
        
        volume_cols = [plant + " Mobilized (mn kWh)" for plant in power_plants]
        revenue_cols = [plant + " Revenue (VNDbn)" for plant in power_plants]
        asp_cols = [plant + " ASP (VND/kWh)" for plant in power_plants]
        
        return {
            'volume': volume_cols,
            'revenue': revenue_cols,
            'asp': asp_cols,
            'subcategories': power_plants
        }
    else:
        # Other companies: REE, GEG, PC1, HDG
        # Look for volume, revenue, and ASP columns
        volume_cols = [col for col in df.columns if 'Volume' in col and 'mn kWh' in col]
        revenue_cols = [col for col in df.columns if 'Revenue' in col and 'VNDbn' in col]
        asp_cols = [col for col in df.columns if 'ASP' in col and 'VND/kWh' in col]
        
        # Extract subcategories (e.g., Solar, Wind, Hydro, Thermal)
        subcategories = []
        for col in volume_cols:
            subcategory = col.split(' Volume')[0]
            subcategories.append(subcategory)
        
        # For REE, remove Thermal from revenue and ASP options
        if company_name.upper() == "REE":
            revenue_cols = [col for col in revenue_cols if 'Thermal' not in col]
            asp_cols = [col for col in asp_cols if 'Thermal' not in col]
        
        return {
            'volume': volume_cols,
            'revenue': revenue_cols,
            'asp': asp_cols,
            'subcategories': subcategories
        }

def get_revenue_asp_subcategories(company_name, df):
    """Get subcategories for revenue and ASP charts (excludes Thermal for REE)"""
    columns_info = get_company_columns(company_name, df)
    
    if company_name.upper() == "POW":
        return columns_info['subcategories']
    elif company_name.upper() == "REE":
        # For REE, exclude Thermal from revenue/ASP subcategories
        return [sub for sub in columns_info['subcategories'] if sub != 'Thermal']
    else:
        return columns_info['subcategories']

def has_data_for_period(df, columns, period_filter=None):
    """Check if there's any non-zero data for the specified columns and period"""
    if period_filter:
        filtered_df = df[period_filter]
    else:
        filtered_df = df
    
    for col in columns:
        if col in filtered_df.columns:
            if filtered_df[col].notna().any() and (filtered_df[col] > 0).any():
                return True
    return False

def create_volume_chart(df, company_name, period, growth_type, selected_subcategories=None):
    """Create volume bar chart for a company"""
    try:
        columns_info = get_company_columns(company_name, df)
        volume_cols = columns_info['volume']
        subcategories = columns_info['subcategories']
        
        # Filter columns based on selected subcategories
        if selected_subcategories:
            if company_name.upper() == "POW":
                selected_volume_cols = [plant + " Mobilized (mn kWh)" for plant in selected_subcategories]
            else:
                selected_volume_cols = [col for col in volume_cols if any(sub in col for sub in selected_subcategories)]
        else:
            selected_volume_cols = volume_cols
        
        # Filter and aggregate data based on period
        if period == "Monthly":
            # Only available for POW
            filtered_df = df[['Date'] + selected_volume_cols].copy()
        elif period == "Quarterly":
            # Data is already quarterly, just select needed columns
            filtered_df = df[['Date', 'Year', 'Quarter'] + selected_volume_cols].copy()
        elif period == "Semi-annually":
            # Aggregate quarterly data to semi-annual
            filtered_df = df.groupby(['Year', 'Half'])[selected_volume_cols].sum().reset_index()
            filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6-3:02d}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])  # H1->Mar, H2->Sep
        else:  # Annually
            # Aggregate quarterly data to annual
            filtered_df = df.groupby('Year')[selected_volume_cols].sum().reset_index()
            filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in filtered_df['Year']])
        
        # Calculate total volume
        filtered_df['Total_Volume'] = filtered_df[selected_volume_cols].sum(axis=1)
        
        # Define company-specific start dates for growth lines
        growth_start_dates = {
            "REE": pd.to_datetime("2022-02-15"),   # 1Q2022
            "GEG": pd.to_datetime("2022-02-15"),   # 1Q2022
            "PC1": pd.to_datetime("2024-02-15"),   # 1Q2024
            "HDG": pd.to_datetime("2024-02-15"),   # 1Q2024
            "POW": pd.to_datetime("2019-01-01")    # Jan-2019
        }
        
        # Check if data meets company-specific threshold and calculate growth accordingly
        required_start_date = growth_start_dates.get(company_name.upper(), pd.to_datetime("1900-01-01"))
        has_threshold_data = (filtered_df['Date'] >= required_start_date).any()
        
        # Calculate growth only when there's sufficient data AND meets date threshold
        if growth_type == "Year-over-Year (YoY)":
            periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
            required_periods = periods_map[period]
            
            # Only calculate growth if we have enough periods for YoY comparison AND meets date threshold
            if len(filtered_df) > required_periods and has_threshold_data:
                # Calculate growth for all data
                filtered_df['Volume_Growth'] = calculate_yoy_growth(filtered_df, 'Total_Volume', required_periods)
                
                # Set growth to NaN for periods before the threshold date
                pre_threshold_mask = filtered_df['Date'] < required_start_date
                filtered_df.loc[pre_threshold_mask, 'Volume_Growth'] = None
                
                growth_title = "YoY Growth (%)"
                show_growth = True
            else:
                show_growth = False
        else:
            # For YTD, need at least 2 periods in the same year AND meets date threshold
            if len(filtered_df) >= 2 and has_threshold_data:
                # Calculate growth for all data
                filtered_df['Volume_Growth'] = calculate_ytd_growth(filtered_df, 'Total_Volume', 'Date', period)
                
                # Set growth to NaN for periods before the threshold date
                pre_threshold_mask = filtered_df['Date'] < required_start_date
                filtered_df.loc[pre_threshold_mask, 'Volume_Growth'] = None
                
                growth_title = "YTD Growth (%)"
                show_growth = True
            else:
                show_growth = False
        
        # Create chart with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Create x-axis labels
        if period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in filtered_df['Date']]
        elif period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in filtered_df['Date']]
        elif period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in filtered_df['Date']]
        
        # Add stacked bars for each subcategory
        colors = ['#0C4130', '#08C179', '#D3BB96', '#B78D51', '#C0C1C2', '#97999B', '#FF6B6B', '#4ECDC4']
        for i, col in enumerate(selected_volume_cols):
            if company_name.upper() == "POW":
                name = col.replace(" Mobilized (mn kWh)", "")
            else:
                name = col.replace(" Volume (mn kWh)", "")
            
            fig.add_trace(
                go.Bar(
                    name=name,
                    x=x_labels,
                    y=filtered_df[col],
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"{name}<br>%{{x}}<br>Volume: %{{y}} mn kWh<extra></extra>"
                ),
                secondary_y=False
            )
        
        # Add growth line only if we have sufficient data
        if show_growth:
            fig.add_trace(
                go.Scatter(
                    name=growth_title,
                    x=x_labels,
                    y=filtered_df['Volume_Growth'],
                    mode='lines+markers',
                    line=dict(color='red', width=2),
                    marker=dict(size=4),
                    hovertemplate=f"{growth_title}<br>%{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>"
                ),
                secondary_y=True
            )
        
        # Update layout
        title_suffix = f" with {growth_title}" if show_growth else ""
        fig.update_layout(
            title=f'{period} {company_name} Volume{title_suffix}',
            barmode='stack',
            hovermode='x unified',
            showlegend=True,
            height=600
        )
        
        if show_growth:
            fig.update_yaxes(title_text="Volume (mn kWh)", secondary_y=False)
            fig.update_yaxes(title_text=growth_title, secondary_y=True)
            fig.update_xaxes(title_text="Date")
            
            # Remove secondary y-axis gridlines
            fig = update_chart_layout_with_no_secondary_grid(fig)
        else:
            fig.update_yaxes(title_text="Volume (mn kWh)")
            fig.update_xaxes(title_text="Date")
        
        return fig, filtered_df, x_labels
        
    except Exception as e:
        st.error(f"Error creating volume chart for {company_name}: {e}")
        return None, None, None

def create_revenue_chart(df, company_name, period, selected_subcategories=None, growth_type="Year-over-Year (YoY)"):
    """Create revenue bar chart for a company"""
    try:
        columns_info = get_company_columns(company_name, df)
        revenue_cols = columns_info['revenue']
        subcategories = columns_info['subcategories']
        
        # Handle different selection logic for POW vs other companies
        if company_name.upper() == "POW":
            # POW: single subcategory selection (existing logic)
            if selected_subcategories:
                if selected_subcategories == "POW Total":
                    revenue_col = revenue_cols  # Use all columns for total
                else:
                    revenue_col = [selected_subcategories + " Revenue (VNDbn)"]
            else:
                revenue_col = revenue_cols
        else:
            # REE, GEG, PC1, HDG: multiple subcategory selection for stacked chart
            if selected_subcategories:
                revenue_col = [col for col in revenue_cols if any(sub in col for sub in selected_subcategories)]
            else:
                revenue_col = revenue_cols
        
        # Filter and aggregate data based on period
        if period == "Monthly":
            # Only available for POW
            filtered_df = df[['Date'] + revenue_col].copy()
        elif period == "Quarterly":
            # Data is already quarterly, just select needed columns
            filtered_df = df[['Date', 'Year', 'Quarter'] + revenue_col].copy()
        elif period == "Semi-annually":
            # Aggregate quarterly data to semi-annual
            filtered_df = df.groupby(['Year', 'Half'])[revenue_col].sum().reset_index()
            filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6-3:02d}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])  # H1->Mar, H2->Sep
        else:  # Annually
            # Aggregate quarterly data to annual
            filtered_df = df.groupby('Year')[revenue_col].sum().reset_index()
            filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in filtered_df['Year']])
        
        # Create x-axis labels
        if period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in filtered_df['Date']]
        elif period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in filtered_df['Date']]
        elif period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in filtered_df['Date']]
        
        # Create chart
        if company_name.upper() == "POW":
            # POW: single bar chart (existing behavior)
            fig = go.Figure()
            
            if len(revenue_col) > 1:
                filtered_df['Total_Revenue'] = filtered_df[revenue_col].sum(axis=1)
                revenue_column = 'Total_Revenue'
            else:
                revenue_column = revenue_col[0]
            
            # Check if there's any data to display
            if filtered_df[revenue_column].notna().any() and (filtered_df[revenue_column] > 0).any():
                fig.add_trace(
                    go.Bar(
                        name="Revenue",
                        x=x_labels,
                        y=filtered_df[revenue_column],
                        marker_color='#08C179',
                        hovertemplate=f"Revenue<br>%{{x}}<br>Revenue: %{{y}} VND bn<extra></extra>"
                    )
                )
            else:
                # Return None if no data to display
                return None, None, None
                
            title_suffix = f" - {selected_subcategories}" if selected_subcategories and selected_subcategories != "POW Total" else ""
            show_legend = False
        else:
            # REE, GEG, PC1, HDG: stacked bar chart with growth line
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Calculate total revenue for growth calculation
            filtered_df['Total_Revenue'] = filtered_df[revenue_col].sum(axis=1)
            
            # Define company-specific start dates for growth lines
            growth_start_dates = {
                "REE": pd.to_datetime("2022-02-15"),   # 1Q2022
                "GEG": pd.to_datetime("2022-02-15"),   # 1Q2022
                "PC1": pd.to_datetime("2024-02-15"),   # 1Q2024
                "HDG": pd.to_datetime("2024-02-15"),   # 1Q2024
                "POW": pd.to_datetime("2019-01-01")    # Jan-2019
            }
            
            # Check if data meets company-specific threshold and calculate growth accordingly
            required_start_date = growth_start_dates.get(company_name.upper(), pd.to_datetime("1900-01-01"))
            has_threshold_data = (filtered_df['Date'] >= required_start_date).any()
            
            # Calculate growth only when there's sufficient data AND meets date threshold
            if growth_type == "Year-over-Year (YoY)":
                periods_map = {"Quarterly": 4, "Semi-annually": 2, "Annually": 1}
                required_periods = periods_map[period]
                
                # Only calculate growth if we have enough periods for YoY comparison AND meets date threshold
                if len(filtered_df) > required_periods and has_threshold_data:
                    # Calculate growth for all data
                    filtered_df['Revenue_Growth'] = calculate_yoy_growth(filtered_df, 'Total_Revenue', required_periods)
                    
                    # Set growth to NaN for periods before the threshold date
                    pre_threshold_mask = filtered_df['Date'] < required_start_date
                    filtered_df.loc[pre_threshold_mask, 'Revenue_Growth'] = None
                    
                    growth_title = "YoY Growth (%)"
                    show_growth = True
                else:
                    show_growth = False
            else:
                # For YTD, need at least 2 periods in the same year AND meets date threshold
                if len(filtered_df) >= 2 and has_threshold_data:
                    # Calculate growth for all data
                    filtered_df['Revenue_Growth'] = calculate_ytd_growth(filtered_df, 'Total_Revenue', 'Date', period)
                    
                    # Set growth to NaN for periods before the threshold date
                    pre_threshold_mask = filtered_df['Date'] < required_start_date
                    filtered_df.loc[pre_threshold_mask, 'Revenue_Growth'] = None
                    
                    growth_title = "YTD Growth (%)"
                    show_growth = True
                else:
                    show_growth = False
            
            colors = ['#0C4130', '#08C179', '#D3BB96', '#B78D51', '#C0C1C2', '#97999B', '#FF6B6B', '#4ECDC4']
            for i, col in enumerate(revenue_col):
                # Extract subcategory name from column (e.g., "Solar Revenue (VNDbn)" -> "Solar")
                name = col.replace(" Revenue (VNDbn)", "")
                
                fig.add_trace(
                    go.Bar(
                        name=name,
                        x=x_labels,
                        y=filtered_df[col],
                        marker_color=colors[i % len(colors)],
                        hovertemplate=f"{name}<br>%{{x}}<br>Revenue: %{{y}} VND bn<extra></extra>"
                    ),
                    secondary_y=False
                )
            
            # Add growth line only if we have sufficient data
            if show_growth:
                fig.add_trace(
                    go.Scatter(
                        name=growth_title,
                        x=x_labels,
                        y=filtered_df['Revenue_Growth'],
                        mode='lines+markers',
                        line=dict(color='red', width=2),
                        marker=dict(size=4),
                        hovertemplate=f"{growth_title}<br>%{{x}}<br>Growth: %{{y:.2f}}%<extra></extra>"
                    ),
                    secondary_y=True
                )
                title_suffix = f" with {growth_title}"
            else:
                title_suffix = ""
                
            show_legend = True
        
        # Update layout
        fig.update_layout(
            title=f'{period} {company_name} Revenue{title_suffix}',
            barmode='stack' if company_name.upper() != "POW" else 'group',
            hovermode='x unified',
            showlegend=show_legend,
            height=500
        )
        
        if company_name.upper() == "POW":
            fig.update_yaxes(title_text="Revenue (VND bn)")
            fig.update_xaxes(title_text="Date")
        else:
            # Non-POW companies
            if show_growth:
                # With secondary y-axis for growth
                fig.update_yaxes(title_text="Revenue (VND bn)", secondary_y=False)
                fig.update_yaxes(title_text=growth_title, secondary_y=True)
                fig.update_xaxes(title_text="Date")
                
                # Remove secondary y-axis gridlines
                fig = update_chart_layout_with_no_secondary_grid(fig)
            else:
                # Without secondary y-axis
                fig.update_yaxes(title_text="Revenue (VND bn)")
                fig.update_xaxes(title_text="Date")
        
        return fig, filtered_df, x_labels
        
    except Exception as e:
        st.error(f"Error creating revenue chart for {company_name}: {e}")
        return None, None, None

def create_asp_chart(df, company_name, period, selected_subcategories=None):
    """Create ASP line chart for a company"""
    try:
        columns_info = get_company_columns(company_name, df)
        asp_cols = columns_info['asp']
        
        # Handle different selection logic for POW vs other companies
        if company_name.upper() == "POW":
            # POW: single subcategory selection (existing logic)
            if selected_subcategories:
                if selected_subcategories == "POW Total":
                    # Calculate weighted average ASP
                    volume_cols = columns_info['volume']
                    revenue_cols = columns_info['revenue']
                    
                    # We'll calculate this in the filtering step
                    asp_col = "Average_ASP"
                else:
                    asp_col = [selected_subcategories + " ASP (VND/kWh)"]
            else:
                asp_col = asp_cols[0] if asp_cols else None
        else:
            # REE, GEG, PC1, HDG: multiple subcategory selection for multiple lines
            if selected_subcategories:
                asp_col = [col for col in asp_cols if any(sub in col for sub in selected_subcategories)]
            else:
                asp_col = asp_cols
        
        if not asp_col:
            st.warning(f"No ASP data available for {company_name}")
            return None, None, None
        
        # Handle POW Total ASP calculation
        if company_name.upper() == "POW" and selected_subcategories == "POW Total":
            # Filter and aggregate data based on period
            volume_cols = columns_info['volume']
            revenue_cols = columns_info['revenue']
            
            if period == "Monthly":
                # Only available for POW
                filtered_df = df[['Date'] + volume_cols + revenue_cols].copy()
            elif period == "Quarterly":
                # Data is already quarterly, just select needed columns
                filtered_df = df[['Date', 'Year', 'Quarter'] + volume_cols + revenue_cols].copy()
            elif period == "Semi-annually":
                # Aggregate quarterly data to semi-annual
                filtered_df = df.groupby(['Year', 'Half'])[volume_cols + revenue_cols].sum().reset_index()
                filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6-3:02d}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])  # H1->Mar, H2->Sep
            else:  # Annually
                # Aggregate quarterly data to annual
                filtered_df = df.groupby('Year')[volume_cols + revenue_cols].sum().reset_index()
                filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in filtered_df['Year']])
            
            # Calculate weighted average ASP
            filtered_df['Total_Revenue'] = filtered_df[revenue_cols].sum(axis=1)
            filtered_df['Total_Volume'] = filtered_df[volume_cols].sum(axis=1)
            filtered_df['Average_ASP'] = (filtered_df['Total_Revenue'] * 1000000000) / (filtered_df['Total_Volume'] * 1000000)  # Convert bn VND to VND and mn kWh to kWh
            
            asp_column = 'Average_ASP'
        else:
            # Regular ASP columns
            if company_name.upper() == "POW":
                # POW: single ASP column
                if isinstance(asp_col, list):
                    asp_col = asp_col[0]
                
                # Filter and aggregate data based on period
                if period == "Monthly":
                    # Only available for POW
                    filtered_df = df[['Date', asp_col]].copy()
                elif period == "Quarterly":
                    # Data is already quarterly, just select needed columns
                    filtered_df = df[['Date', 'Year', 'Quarter', asp_col]].copy()
                elif period == "Semi-annually":
                    # Aggregate quarterly data to semi-annual (average ASP)
                    filtered_df = df.groupby(['Year', 'Half'])[asp_col].mean().reset_index()
                    filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6-3:02d}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])  # H1->Mar, H2->Sep
                else:  # Annually
                    # Aggregate quarterly data to annual (average ASP)
                    filtered_df = df.groupby('Year')[asp_col].mean().reset_index()
                    filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in filtered_df['Year']])
                
                asp_column = asp_col
            else:
                # REE, GEG, PC1, HDG: multiple ASP columns
                # Filter and aggregate data based on period
                if period == "Quarterly":
                    # Data is already quarterly, just select needed columns
                    filtered_df = df[['Date', 'Year', 'Quarter'] + asp_col].copy()
                elif period == "Semi-annually":
                    # Aggregate quarterly data to semi-annual (average ASP)
                    filtered_df = df.groupby(['Year', 'Half'])[asp_col].mean().reset_index()
                    filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6-3:02d}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])  # H1->Mar, H2->Sep
                else:  # Annually
                    # Aggregate quarterly data to annual (average ASP)
                    filtered_df = df.groupby('Year')[asp_col].mean().reset_index()
                    filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in filtered_df['Year']])
                
                asp_column = asp_col
        
        # Create x-axis labels
        if period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in filtered_df['Date']]
        elif period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in filtered_df['Date']]
        elif period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in filtered_df['Date']]
        
        # Create line chart
        fig = go.Figure()
        
        if company_name.upper() == "POW":
            # POW: single line chart (existing behavior)
            # Check if there's any data to display
            if asp_column in filtered_df.columns and filtered_df[asp_column].notna().any() and (filtered_df[asp_column] > 0).any():
                fig.add_trace(
                    go.Scatter(
                        name="ASP",
                        x=x_labels,
                        y=filtered_df[asp_column],
                        mode='lines+markers',
                        line=dict(color='#FF6B6B', width=3),
                        marker=dict(size=6),
                        hovertemplate=f"ASP<br>%{{x}}<br>ASP: %{{y:.0f}} VND/kWh<extra></extra>"
                    )
                )
            else:
                # Return None if no data to display
                return None, None, None
                
            title_suffix = f" - {selected_subcategories}" if selected_subcategories and selected_subcategories != "POW Total" else ""
            show_legend = False
        else:
            # REE, GEG, PC1, HDG: multiple line chart
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            for i, col in enumerate(asp_col):
                # Extract subcategory name from column (e.g., "Solar ASP (VND/kWh)" -> "Solar")
                name = col.replace(" ASP (VND/kWh)", "")
                
                fig.add_trace(
                    go.Scatter(
                        name=name,
                        x=x_labels,
                        y=filtered_df[col],
                        mode='lines+markers',
                        line=dict(color=colors[i % len(colors)], width=3),
                        marker=dict(size=6),
                        hovertemplate=f"{name} ASP<br>%{{x}}<br>ASP: %{{y:.0f}} VND/kWh<extra></extra>"
                    )
                )
            title_suffix = ""
            show_legend = True
        
        # Update layout
        fig.update_layout(
            title=f'{period} {company_name} ASP{title_suffix}',
            hovermode='x unified',
            showlegend=show_legend,
            height=500
        )
        
        fig.update_yaxes(title_text="ASP (VND/kWh)")
        fig.update_xaxes(title_text="Date")
        
        return fig, filtered_df, x_labels
        
    except Exception as e:
        st.error(f"Error creating ASP chart for {company_name}: {e}")
        return None, None, None

def render_company_tab():
    """Render the main Company tab with sub-tabs"""
    st.header("Company Analysis")
    
    # Create sub-tabs
    company_tabs = st.tabs(["REE", "GEG", "PC1", "HDG", "POW"])
    
    companies = ["REE", "GEG", "PC1", "HDG", "POW"]
    
    for i, company in enumerate(companies):
        with company_tabs[i]:
            render_company_subtab(company)

def create_pow_comparison_chart(df, period, selected_plant):
    """Create contracted vs mobilized comparison chart for POW - single plant or total"""
    try:
        if not selected_plant:
            return None, None
        
        if selected_plant == "Total":
            # Calculate total across all plants
            power_plants = ['Ca Mau', 'Nhon Trach 1', 'Nhon Trach 2 (NT2)', 'Hua Na', 'Dak Drinh', 'Vung Ang']
            contracted_cols = [plant + " Contracted (mn kWh)" for plant in power_plants if plant + " Contracted (mn kWh)" in df.columns]
            mobilized_cols = [plant + " Mobilized (mn kWh)" for plant in power_plants if plant + " Mobilized (mn kWh)" in df.columns]
        else:
            # Single plant
            contracted_cols = [selected_plant + " Contracted (mn kWh)"]
            mobilized_cols = [selected_plant + " Mobilized (mn kWh)"]
            
            # Check if columns exist
            contracted_cols = [col for col in contracted_cols if col in df.columns]
            mobilized_cols = [col for col in mobilized_cols if col in df.columns]
        
        if not contracted_cols or not mobilized_cols:
            return None, None
        
        # Filter and aggregate data based on period
        if period == "Monthly":
            filtered_df = df[['Date'] + contracted_cols + mobilized_cols].copy()
        elif period == "Quarterly":
            filtered_df = df[['Date', 'Year', 'Quarter'] + contracted_cols + mobilized_cols].copy()
        elif period == "Semi-annually":
            filtered_df = df.groupby(['Year', 'Half'])[contracted_cols + mobilized_cols].sum().reset_index()
            filtered_df['Date'] = pd.to_datetime([f"{y}-{h*6-3:02d}-01" for y, h in zip(filtered_df['Year'], filtered_df['Half'])])
        else:  # Annually
            filtered_df = df.groupby('Year')[contracted_cols + mobilized_cols].sum().reset_index()
            filtered_df['Date'] = pd.to_datetime([f"{int(y)}-01-01" for y in filtered_df['Year']])
        
        # Create x-axis labels
        if period == "Monthly":
            x_labels = [d.strftime('%b %Y') for d in filtered_df['Date']]
        elif period == "Quarterly":
            x_labels = [f"Q{d.quarter} {d.year}" for d in filtered_df['Date']]
        elif period == "Semi-annually":
            x_labels = [f"H{((d.month-1)//6)+1} {d.year}" for d in filtered_df['Date']]
        else:
            x_labels = [str(int(d.year)) for d in filtered_df['Date']]
        
        # Calculate data for the chart
        if selected_plant == "Total":
            contracted_data = filtered_df[contracted_cols].sum(axis=1)
            mobilized_data = filtered_df[mobilized_cols].sum(axis=1)
        else:
            contracted_data = filtered_df[contracted_cols[0]] if contracted_cols else pd.Series()
            mobilized_data = filtered_df[mobilized_cols[0]] if mobilized_cols else pd.Series()
        
        # Calculate utilization rate (avoid division by zero)
        utilization_rate = pd.Series(0, index=contracted_data.index)
        mask = contracted_data > 0
        utilization_rate[mask] = (mobilized_data[mask] / contracted_data[mask] * 100).round(2)
        
        # Calculate total volume for potential growth calculation
        total_volume = mobilized_data
        
        # Check if there's any data to display
        if contracted_data.sum() == 0 and mobilized_data.sum() == 0:
            return None, None
        
        # Create simple 2-column bar chart with utilization rate
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add contracted volume bar
        fig.add_trace(
            go.Bar(
                name="Contracted",
                x=x_labels,
                y=contracted_data.values,
                marker_color='lightblue',
                hovertemplate="Contracted<br>%{x}<br>Volume: %{y:.2f} mn kWh<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add mobilized volume bar
        fig.add_trace(
            go.Bar(
                name="Mobilized",
                x=x_labels,
                y=mobilized_data.values,
                marker_color='darkblue',
                hovertemplate="Mobilized<br>%{x}<br>Volume: %{y:.2f} mn kWh<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add utilization rate line
        fig.add_trace(
            go.Scatter(
                name="Utilization Rate",
                x=x_labels,
                y=utilization_rate.values,
                mode='lines+markers',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                hovertemplate="Utilization Rate<br>%{x}<br>Rate: %{y:.2f}%<extra></extra>"
            ),
            secondary_y=True
        )
        
        # Update layout
        plant_title = selected_plant if selected_plant != "Total" else "All Plants"
        fig.update_layout(
            title=f'{period} POW Contracted vs Mobilized Volume - {plant_title}',
            barmode='group',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        fig.update_yaxes(title_text="Volume (mn kWh)", secondary_y=False)
        fig.update_yaxes(title_text="Utilization Rate (%)", secondary_y=True)
        fig.update_xaxes(title_text="Date")
        
        # Remove secondary y-axis gridlines
        fig = update_chart_layout_with_no_secondary_grid(fig)
        
        # Store data for return
        filtered_df['Contracted'] = contracted_data
        filtered_df['Mobilized'] = mobilized_data
        filtered_df['Utilization_Rate'] = utilization_rate
        
        return fig, filtered_df
        
    except Exception as e:
        st.error(f"Error creating POW comparison chart: {e}")
        return None, None

def render_company_subtab(company_name):
    """Render individual company sub-tab"""
    st.subheader(f"{company_name} Analysis")
    
    # Load company data
    df, has_data = load_company_data(company_name)
    
    if not has_data or df is None:
        st.error(f"❌ No data available for {company_name}")
        return
    
    # Get company-specific columns
    columns_info = get_company_columns(company_name, df)
    subcategories = columns_info['subcategories']
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        # Different time period options based on company
        if company_name.upper() == "POW":
            period_options = ["Monthly", "Quarterly", "Semi-annually", "Annually"]
        else:
            period_options = ["Quarterly", "Semi-annually", "Annually"]
        
        period = st.selectbox(
            "Select Time Period:",
            period_options,
            key=f"{company_name}_period"
        )
    
    with col2:
        growth_type = st.selectbox(
            "Select Growth Type:",
            ["Year-over-Year (YoY)", "Year-to-Date (YTD)"],
            key=f"{company_name}_growth_type"
        )
    
    # Volume Chart
    st.subheader("📊 Volume Analysis")
    
    if company_name.upper() == "POW":
        # POW has power plants
        selected_subcategories = st.multiselect(
            "Select Power Plants:",
            subcategories,
            default=subcategories,
            key=f"{company_name}_volume_subcategories"
        )
    else:
        # Other companies have energy types
        selected_subcategories = st.multiselect(
            "Select Energy Types:",
            subcategories,
            default=subcategories,
            key=f"{company_name}_volume_subcategories"
        )
    
    if selected_subcategories:
        volume_fig, volume_df, volume_labels = create_volume_chart(df, company_name, period, growth_type, selected_subcategories)
        if volume_fig:
            st.plotly_chart(volume_fig, use_container_width=True)
            
            # Download data
            st.subheader("📥 Download Volume Data")
            add_download_buttons(volume_df, f"{company_name}_volume_{period.lower()}")
    else:
        st.warning("Please select at least one subcategory for volume analysis.")
    
    # POW-specific Contracted vs Mobilized Comparison
    if company_name.upper() == "POW":
        st.subheader("🔄 Contracted vs Mobilized Comparison")
        
        # Controls for comparison chart (same format as revenue/ASP)
        col1, col2 = st.columns(2)
        
        with col1:
            comp_period = st.selectbox(
                "Select Time Period:",
                ["Monthly", "Quarterly", "Semi-annually", "Annually"],
                key=f"{company_name}_comp_period"
            )
        
        with col2:
            # Empty column for consistent layout
            st.write("")
        
        # Plant selection (single selectbox with Total option)
        plant_options = ["Total"] + subcategories
        selected_comp_plant = st.selectbox(
            "Select Power Plant:",
            plant_options,
            index=0,  # Default to "Total"
            key=f"{company_name}_comp_plant"
        )
        
        if selected_comp_plant:
            comp_fig, comp_df = create_pow_comparison_chart(df, comp_period, selected_comp_plant)
            if comp_fig:
                st.plotly_chart(comp_fig, use_container_width=True)
                
                # Download comparison data
                st.subheader("📥 Download Comparison Data")
                plant_suffix = selected_comp_plant.lower().replace(' ', '_').replace('(', '').replace(')', '')
                add_download_buttons(comp_df, f"{company_name}_comparison_{plant_suffix}_{comp_period.lower()}")
        else:
            st.warning("Please select a power plant for comparison analysis.")
    
    # Revenue Chart
    st.subheader("💰 Revenue Analysis")
    
    if company_name.upper() == "POW":
        # POW: single selection
        revenue_options = subcategories + ["POW Total"]
        selected_revenue_subcategory = st.selectbox(
            "Select Category for Revenue:",
            revenue_options,
            key=f"{company_name}_revenue_subcategory"
        )
    else:
        # REE, GEG, PC1, HDG: multiple selection for stacked chart
        revenue_subcategories = get_revenue_asp_subcategories(company_name, df)
        selected_revenue_subcategory = st.multiselect(
            "Select Categories for Revenue:",
            revenue_subcategories,
            default=revenue_subcategories,
            key=f"{company_name}_revenue_subcategories"
        )
    
    if selected_revenue_subcategory:
        revenue_fig, revenue_df, revenue_labels = create_revenue_chart(df, company_name, period, selected_revenue_subcategory, growth_type)
        if revenue_fig:
            st.plotly_chart(revenue_fig, use_container_width=True)
            
            # Download data
            st.subheader("📥 Download Revenue Data")
            add_download_buttons(revenue_df, f"{company_name}_revenue_{period.lower()}")
    else:
        if company_name.upper() != "POW":
            st.warning("Please select at least one category for revenue analysis.")
    
    # ASP Chart
    st.subheader("📈 ASP Analysis")
    
    if company_name.upper() == "POW":
        # POW: single selection (same as revenue options)
        asp_options = subcategories + ["POW Total"]
        selected_asp_subcategory = st.selectbox(
            "Select Category for ASP:",
            asp_options,
            key=f"{company_name}_asp_subcategory"
        )
    else:
        # REE, GEG, PC1, HDG: multiple selection for multiple lines
        asp_subcategories = get_revenue_asp_subcategories(company_name, df)
        selected_asp_subcategory = st.multiselect(
            "Select Categories for ASP:",
            asp_subcategories,
            default=asp_subcategories,
            key=f"{company_name}_asp_subcategories"
        )
    
    if selected_asp_subcategory:
        asp_fig, asp_df, asp_labels = create_asp_chart(df, company_name, period, selected_asp_subcategory)
        if asp_fig:
            st.plotly_chart(asp_fig, use_container_width=True)
            
            # Download data
            st.subheader("📥 Download ASP Data")
            add_download_buttons(asp_df, f"{company_name}_asp_{period.lower()}")
    else:
        if company_name.upper() != "POW":
            st.warning("Please select at least one category for ASP analysis.")
