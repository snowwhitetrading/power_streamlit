import pandas as pd
import numpy as np

def process_data():
    # Read the data files
    print("Reading data files...")
    volume_df = pd.read_csv('phutai_data.csv')
    price_df = pd.read_csv('gia_bien_data.csv')
    
    # Print dataframe information
    print("\nVolume DataFrame columns:")
    print(volume_df.columns.tolist())
    print("\nPrice DataFrame columns:")
    print(price_df.columns.tolist())
    
    # Convert datetime columns
    volume_df['thoiGian'] = pd.to_datetime(volume_df['thoiGian'])
    price_df['thoiGian'] = pd.to_datetime(price_df['thoiGian'])
    
    # Remove MB, MT, MN columns from volume data
    volume_df = volume_df[['thoiGian', 'congSuatHT']]
    
    # Merge the dataframes on datetime
    print("Merging datasets...")
    merged_df = pd.merge(volume_df, 
                        price_df[['thoiGian', 'giaBienHT']], 
                        on='thoiGian', 
                        how='inner')
    
    # Remove rows where price < 50
    print("Filtering out prices < 50...")
    merged_df = merged_df[merged_df['giaBienHT'] >= 50]
    
    # Add date column for grouping
    merged_df['date'] = merged_df['thoiGian'].dt.date
    
    # Calculate volume-weighted average price for each day
    print("Calculating volume-weighted average prices...")
    result_df = (merged_df.groupby('date')
                .apply(lambda x: np.average(x['giaBienHT'], 
                                         weights=x['congSuatHT']))
                .reset_index())
    result_df.columns = ['date', 'weighted_avg_price']
    
    # Add volume sum for reference
    volume_sums = (merged_df.groupby('date')['congSuatHT']
                  .sum()
                  .reset_index())
    result_df = result_df.merge(volume_sums, on='date')
    
    # Sort by date
    result_df = result_df.sort_values('date')
    
    # Save the results
    output_file = 'weighted_average_prices.csv'
    result_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print("\nFirst few rows of results:")
    print(result_df.head())
    
    return result_df

if __name__ == "__main__":
    try:
        result = process_data()
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
