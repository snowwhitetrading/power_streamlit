import pandas as pd

# Read the CSV file
df = pd.read_csv('Monthly_Power_Volume.csv')

# Save to Excel
df.to_excel('Monthly_Power_Volume.xlsx', index=False)
print("Excel file updated successfully!")
