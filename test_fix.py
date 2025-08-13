import pandas as pd

# Test the data loading fix
df = pd.read_csv('power_volume.csv')
print('Original data types:')
print(df.dtypes)
print('\nFirst row original:')
print(df.iloc[0])

# Apply the fix
numeric_columns = ['Hydro', 'Coals', 'Gas', 'Renewables', 'Import & Diesel']
for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

print('\nAfter cleaning - data types:')
print(df.dtypes)
print('\nFirst row after cleaning:')
print(df.iloc[0])

# Test the calculation
df['Total'] = df[numeric_columns].sum(axis=1)
print('\nTotal calculation works:')
print(df['Total'].head())

# Test pct_change
df['Total_Growth'] = df['Total'].pct_change() * 100
print('\nGrowth calculation works:')
print(df['Total_Growth'].head())
