import requests
import pandas as pd
import time
import os

def fetch_binance_klines(symbol='ETHUSDT', interval='1h', total_records=60000, filename='extracted_dataset.csv'):
    """
    Fetches historical K-line data from Binance and saves it to a CSV.
    Tries multiple mirror endpoints to bypass regional restrictions (e.g., in GitHub Actions).
    """
    endpoints = [
        "https://api.binance.com/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines"
    ]
    
    all_data = []
    last_time = None
    limit = 1000
    batches_needed = (total_records // limit) + (1 if total_records % limit > 0 else 0)
    
    print(f"Starting extraction for {symbol} ({interval})...")
    
    current_endpoint_idx = 0
    
    for i in range(batches_needed):
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        # If we already have data, fetch records older than the oldest one we have
        if last_time:
            params['endTime'] = last_time - 1

        success = False
        attempts = 0
        
        while not success and attempts < len(endpoints):
            url = endpoints[current_endpoint_idx]
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    print("No more data available on Binance servers.")
                    break
                    
                all_data.extend(data)
                
                # The first element of the first list in 'data' is the 'open_time'
                last_time = data[0][0]
                
                print(f"Progress: {len(all_data)}/{total_records} records collected (via {url})...")
                
                # Small sleep to respect API rate limits
                time.sleep(0.1)
                success = True
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 451:
                    print(f"Endpoint {url} blocked (451). Trying next mirror...")
                    current_endpoint_idx = (current_endpoint_idx + 1) % len(endpoints)
                    attempts += 1
                else:
                    print(f"Error fetching batch {i+1}: {e}")
                    break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        
        if not success and attempts >= len(endpoints):
            print("All Binance endpoints are blocked or unavailable.")
            break

    # Define columns to match your btc_dataset.csv
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    # Create DataFrame
    df = pd.DataFrame(all_data, columns=columns)
    
    # Sort by time (data was collected backwards, so we flip it)
    df = df.sort_values('open_time').reset_index(drop=True)
    
    # Keep exactly the number of records requested (from the most recent end)
    df = df.tail(total_records)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Successfully saved {len(df)} records to {filename}")
    return df

# --- Execution ---
# This will fetch the latest 60,000 hours of BTC/USDT data
if __name__ == "__main__":
    # You can change 'BTCUSDT' to 'ETHUSDT', 'SOLUSDT', etc.
    # Intervals: '1m', '5m', '15m', '1h', '4h', '1d'
    dataset = fetch_binance_klines(symbol='ETHUSDT', interval='1h', total_records=60000)