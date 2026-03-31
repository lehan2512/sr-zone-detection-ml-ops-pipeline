import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_stats(df, label):
    stats = {
        "mean_close": df['close'].mean(),
        "std_close": df['close'].std(),
        "var_close": df['close'].var(),
        "mean_vol": df['volume'].mean(),
        "std_vol": df['volume'].std(),
        "var_vol": df['volume'].var(),
        "rows": len(df),
        "nulls": df.isnull().sum().sum()
    }
    logger.info(f"--- Stats for {label} ---")
    for k, v in stats.items():
        logger.info(f"{k}: {v:.4f}")
    return stats

def main():
    path = Path("datasets/BTCUSDT_raw.csv")
    if not path.exists():
        logger.error(f"File not found: {path}")
        return

    df = pd.read_csv(path)
    
    # 1. Initial Stats
    check_stats(df, "Initial Raw Data")
    
    # 2. Handle Nulls
    df['close'] = df['close'].ffill()
    df['volume'] = df['volume'].fillna(0)
    df.dropna(subset=['close'], inplace=True)
    check_stats(df, "After Null Handling")
    
    # 3. RVT Filter (Rolling Volume Threshold)
    # Using 75th percentile as in _filter_noise
    window = 20
    thresholds = df['volume'].rolling(window=window).quantile(0.75).shift(1)
    
    df_filtered = df[df['volume'] >= thresholds].copy()
    check_stats(df_filtered, "After RVT Filter (75th percentile)")

if __name__ == "__main__":
    main()
