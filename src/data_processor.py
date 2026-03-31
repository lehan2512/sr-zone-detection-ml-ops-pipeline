import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional, Union
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Use a module-level logger
logger = logging.getLogger(__name__)

class SREngineer:
    """
    Engineers features and targets for Support/Resistance detection.
    
    Attributes:
        config (dict): Configuration dictionary for hyperparameters.
    """
    def __init__(self, config: Optional[dict] = None):
        # Default parameters if no config is provided
        self.params = {
            "rsi_period": 14,
            "atr_period": 14,
            "vol_ma_period": 20,
            "extrema_window": 20,
            "min_clusters": 2,
            "max_clusters": 8,
            "tolerance": 0.01
        }
        if config:
            self.params.update(config)
            
        self.support_centers: List[float] = []
        self.resistance_centers: List[float] = []

    def process_pipeline(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Executes the end-to-end data processing pipeline (Sanitization + Feature Engineering)."""
        path = Path(filepath)
        try:
            logger.info(f"Starting data processing pipeline for {path}")
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found at: {path}")

            df = pd.read_csv(path)
            
            # Phase 1: Sanitization
            df = self._sanitize_data(df)
            
            # Phase 2: Feature Engineering (Rolling windows work here)
            df = self._engineer_features(df)
            
            # Phase 3: RVT Filter & Stats Analysis
            # We calculate stats before and after the filter to prove quality improvement
            self._log_quality_metrics(df, "Pre-RVT Filter")
            
            df = self._apply_rvt_filter(df)
            
            self._log_quality_metrics(df, "Post-RVT Filter")
            
            logger.info("Data processing pipeline completed successfully.")
            return df
            
        except Exception as e:
            logger.error(f"Pipeline failure: {e}")
            raise

    def _sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data artifacts and handle nulls/outliers."""
        df = df.copy()
        initial_len = len(df)
        
        # 1. Filter irrelevant columns
        # We only need: (open, high, low, close, volume, number_of_trades)
        keep_cols = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades']
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols]
            
        # 2. Handle Null Values
        # Forward fill price data, fill volume and trades with 0
        price_cols = ['open', 'high', 'low', 'close']
        df[price_cols] = df[price_cols].ffill()
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        if 'number_of_trades' in df.columns:
            df['number_of_trades'] = df['number_of_trades'].fillna(0)
        
        # Drop rows where we still have NaNs in price (e.g., at the very beginning)
        df.dropna(subset=price_cols, inplace=True)
            
        # 3. Handle Price Outliers (Avoid touching volume/trades as per instructions)
        # Using a conservative approach: remove rows where high < low or price <= 0
        df = df[(df['high'] >= df['low']) & (df['close'] > 0)]
        
        # 4. Remove dead zones (volume = 0)
        if 'volume' in df.columns:
            df = df[df['volume'] > 0]
        
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Sanitization complete. Removed {initial_len - len(df)} rows.")
        return df

    def _apply_rvt_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Rolling Volume Threshold (RVT) filter to improve data quality.
        Keeps bars where volume is in the upper quartile of the rolling window.
        """
        p = self.params
        # Compute threshold (75th percentile)
        thresholds = df['volume'].rolling(window=p['vol_ma_period']).quantile(0.75).shift(1)
        
        # Filter: keep high-volume bars that are 'more reliable'
        df_filtered = df[df['volume'] >= thresholds].copy()
        df_filtered.reset_index(drop=True, inplace=True)
        
        return df_filtered

    def _log_quality_metrics(self, df: pd.DataFrame, label: str):
        """Calculates and logs statistics to prove data quality improvement."""
        metrics = {
            "Mean Close": df['close'].mean(),
            "Std Close": df['close'].std(),
            "Var Close": df['close'].var(),
            "Mean Volume": df['volume'].mean(),
            "Std Volume": df['volume'].std(),
            "Var Volume": df['volume'].var(),
            "Count": len(df)
        }
        
        logger.info(f"--- Data Quality Metrics: {label} ---")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for ML model."""
        p = self.params
        
        # 1. Volume MA
        df['vol_ma'] = df['volume'].rolling(window=p['vol_ma_period']).mean()
        
        # 2. RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 3. ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=p['atr_period']).mean()
        
        # 4. Lags & Momentum
        df['rsi_lag1'] = df['rsi'].shift(1)
        df['vol_ma_lag1'] = df['vol_ma'].shift(1)
        df['price_roc'] = df['close'].pct_change(periods=3) * 100
        
        return df.dropna().reset_index(drop=True)

    def _find_optimal_clusters(self, data: np.ndarray, zone_name: str = "Zone") -> List[float]:
        """Dynamically find clusters using Silhouette Score."""
        min_k = self.params['min_clusters']
        max_k = min(self.params['max_clusters'], len(data) - 1)

        if len(data) <= min_k:
            logger.warning(f"Insufficient data for {zone_name} clustering.")
            return []

        best_score, best_centers, best_k = -1, [], 0
        X = data.reshape(-1, 1)

        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score, best_centers, best_k = score, km.cluster_centers_.flatten(), k

        logger.info(f"Optimal K for {zone_name}: {best_k} (Score: {best_score:.4f})")
        return list(best_centers)

    def fit_clusters(self, train_df: pd.DataFrame) -> None:
        """Identify S/R levels using ONLY training data to prevent leakage."""
        win = self.params['extrema_window']
        prices = train_df['close'].values
        
        # Extrema Detection
        peaks = argrelextrema(prices, np.greater, order=win)[0]
        troughs = argrelextrema(prices, np.less, order=win)[0]
        
        # Volume Filter
        valid_peaks = self._filter_noise(peaks, train_df)
        valid_troughs = self._filter_noise(troughs, train_df)
        
        logger.info(f"Volume filter retained {len(valid_peaks)} peaks and {len(valid_troughs)} troughs from training data.")
        
        # Clustering
        self.resistance_centers = self._find_optimal_clusters(valid_peaks, "Resistance")
        self.support_centers = self._find_optimal_clusters(valid_troughs, "Support")

    def label_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Labels a dataframe based on proximity to previously fitted S/R centers."""
        df = df.copy()
        tol = self.params['tolerance']
        df['target'] = 0 
        
        if not self.support_centers and not self.resistance_centers:
            logger.warning("No S/R centers found. Targets will all be 0.")
            return df

        for center in self.support_centers:
            df.loc[df['close'].between(center*(1-tol), center*(1+tol)), 'target'] = 1
        for center in self.resistance_centers:
            df.loc[df['close'].between(center*(1-tol), center*(1+tol)), 'target'] = 2
            
        return df

    def _filter_noise(self, indices: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """Vectorized noise filter based on volume thresholding."""
        # Compute the 75th percentile threshold for volume across the entire DF once
        # shift(1) ensures we look at the 'previous periods' to prevent data leakage
        thresholds = df['volume'].rolling(window=self.params['vol_ma_period']).quantile(0.75).shift(1)
        
        # Vectorized filter: Keep indices where volume >= threshold
        valid_mask = df['volume'].iloc[indices] >= thresholds.iloc[indices]
        
        return df['close'].iloc[indices][valid_mask].values