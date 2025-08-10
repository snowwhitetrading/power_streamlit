import pandas as pd

# Read the Excel file
df = pd.read_excel('Monthly_Power_Volume.xlsx')

# Save to CSV
df.to_csv('Monthly_Power_Volume.csv', index=False)
print("CSV file updated successfully!")
