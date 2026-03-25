import logging
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
    
def clean_btc_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"File not found: '{file_path}'")
        return None

    original_count = len(df)
    logger.info(f"Original record count: {original_count}")

    # ---Data Preprocessing---
    # Remove extra columns
    required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df = df[required_columns]

    # Handle missing values
    prev_len = len(df)
    df = df.dropna()
    dropped_rows = prev_len - len(df)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows containing missing values.")

    # Normalize numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']

    # Normalize volume
    scaler = MinMaxScaler()
    df['volume_norm'] = scaler.fit_transform(df[['volume']])
    
    # Feature Engineering
    # Calculate  spread to to find the volatility anomalies
    df['spread'] = df['high'] - df['low']
    df['spread_norm'] = scaler.fit_transform(df[['spread']])
    
    # ---Anomaly Detection using DBSCAN---
    X = df[['volume_norm', 'spread_norm']]
    dbscan = DBSCAN(eps=0.05, min_samples=5) 
    df['cluster'] = dbscan.fit_predict(X)

    noise_data = df[df['cluster'] == -1]
    clean_data = df[df['cluster'] != -1]

    removed_count = len(noise_data)
    remaining_count = len(clean_data)

    logger.info(f"Records removed   : {removed_count}")
    logger.info(f"Records remaining : {remaining_count}")

    

    # Drop calculated columns
    final_output = clean_data.drop(columns=['spread', 'spread_norm', 'cluster'])

    output_filename = 'datasets/btc_dataset_cleaned.csv'
    final_output.to_csv(output_filename, index=False)
    logger.info(f"Cleaned dataset saved to: {output_filename}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    clean_btc_data('datasets/btc_dataset.csv')