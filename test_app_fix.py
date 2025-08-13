import pandas as pd
import sys

# Test the load_data function without streamlit cache
def load_data_test():
    df = pd.read_csv('power_volume.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean numeric columns - remove commas and spaces, convert to numeric
    numeric_columns = ['Hydro', 'Coals', 'Gas', 'Renewables', 'Import & Diesel']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['Half'] = (df['Date'].dt.month - 1) // 6 + 1
    return df

try:
    df = load_data_test()
    print('✅ Data loading works correctly')
    
    # Test the problematic calculation
    selected_power_types = ['Gas', 'Hydro', 'Coals']
    filtered_df = df.copy()
    filtered_df['Total'] = filtered_df[selected_power_types].sum(axis=1)
    filtered_df['Total'] = pd.to_numeric(filtered_df['Total'], errors='coerce')
    
    # Test growth calculation
    filtered_df['Total_Growth'] = filtered_df['Total'].pct_change() * 100
    filtered_df['Total_Growth'] = filtered_df['Total_Growth'].fillna(0)
    
    print('✅ Growth calculation works correctly')
    print(f'✅ Data shape: {filtered_df.shape}')
    print(f'✅ Total range: {filtered_df["Total"].min()} - {filtered_df["Total"].max()}')
    
    # Test YoY calculation with periods
    periods_map = {"Monthly": 12, "Quarterly": 4, "Semi-annually": 2, "Annually": 1}
    period = "Monthly"
    filtered_df['YoY_Growth'] = filtered_df['Total'].pct_change(periods=periods_map[period]) * 100
    filtered_df['YoY_Growth'] = filtered_df['YoY_Growth'].fillna(0)
    print('✅ YoY Growth calculation works correctly')
    
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
