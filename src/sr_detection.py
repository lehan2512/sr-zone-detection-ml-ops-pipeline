import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class SRZoneDetector(BaseEstimator):
    def __init__(self, dbscan_eps=0.05, min_samples=5, rf_trees=100, max_k=12):
        self.dbscan_eps = dbscan_eps
        self.min_samples = min_samples
        self.rf_trees = rf_trees
        self.max_k = max_k
        
        self.validator_model = RandomForestClassifier(n_estimators=rf_trees, random_state=42)
        self.is_trained = False
        self.feature_importance_ = None

    def _validate_input(self, df):
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Input dataframe missing required columns: {missing}")

        df = df[required_columns].copy()

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        original_len = len(df)
        df.dropna(inplace=True)

        if len(df) < original_len:
            logger.warning(f"Dropped {original_len - len(df)} rows containing NaN/Inf values.")
            
        if df.empty:
            raise ValueError("Dataframe is empty after cleaning!")
            
        return df
    
    # Data Preprocessing and Cleaning
    def _clean_data(self, df):
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        
        logger.info(f"   [Clean] Starting DBSCAN noise removal on {len(df)} records...")

        # handle missing values
        original_count = len(df)
        clean_df = df[req_cols].dropna()
        dropped_rows = original_count - len(clean_df)
        
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows containing missing values")
        if clean_df.empty:
            raise ValueError("Dataframe is empty after dropping missing values")
        
        # Normalize volume and spread
        scaler = MinMaxScaler()
        vol_norm = scaler.fit_transform(clean_df[['volume']])
        spread = (df['high'] - df['low']).values.reshape(-1, 1)
        spread_norm = scaler.fit_transform(spread)
        
        if np.isnan(vol_norm).any() or np.isnan(spread_norm).any():
            raise ValueError("NaNs generated during scaling. Check for zero variance in volume/price.")
        
        # DBSCAN for anomaly removal (noise)
        X = np.column_stack((vol_norm, spread_norm))
        db = DBSCAN(eps=self.dbscan_eps, min_samples=self.min_samples)
        labels = db.fit_predict(X)
        
        noise_mask = (labels == -1)
        removed_count = np.sum(noise_mask)
        remaining_count = original_count - removed_count

        if removed_count > 0:
            logger.info(f"   [Clean] Removed {removed_count} noise points. Remaining: {remaining_count}")
        else:
            logger.info("   [Clean] No noise points detected by DBSCAN.")
                
        if remaining_count == 0:
            raise ValueError("DBSCAN removed ALL data. dbscan_eps too low")

        return clean_df[~noise_mask]
    
    # K-Means Clustering with Elbow Method
    def _find_optimal_k(self, X, max_k=12):
        logger.info("   [Cluster] Tuning Hyperparameters: Finding optimal k using Elbow Method...")
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            
        # Vector math to find the point furthest from the straight line
        p1 = np.array([k_range[0], inertias[0]])
        p2 = np.array([k_range[-1], inertias[-1]])
        
        best_k = 2
        max_dist = 0

        for i, k in enumerate(k_range):
            p = np.array([k, inertias[i]])

            # Calculate perpendicular distance from point p to line p1-p2
            dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)

            if dist > max_dist:
                max_dist = dist
                best_k = k
        
        logger.info(f"   [Cluster] Optimal k detected: {best_k}")
        return best_k

    def _generate_zones(self, df):
        price_data = df[['close']].values
        best_k = self._find_optimal_k(price_data)
        
        logger.info(f"   [Cluster] Generating {best_k} zones using K-Means...")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        df['sr_cluster'] = kmeans.fit_predict(price_data)
        
        centroids = kmeans.cluster_centers_
        zones = []
        for i, center in enumerate(centroids):
            zones.append({
                "cluster_id": i, 
                "price_level": float(center[0]), 
                "member_count": int(len(df[df['sr_cluster'] == i]))
            })
        
        # Sort zones by price level    
        zones_list = sorted(zones, key=lambda x: x['price_level'])

        return df, zones_list
    
    def _extract_features(self, df, zones):
        if not zones:
            logger.error("No zones found")
            return
        
        logger.info("   [Validate] Extracting features for Random Forest...")

        features = []
        all_counts = [z['member_count'] for z in zones]
        median_touches = np.median(all_counts) if all_counts else 0
        
        for z in zones:
            cid = z['cluster_id']
            cluster_data = df[df['sr_cluster'] == cid]
            
            if len(cluster_data) == 0: 
                continue
            
            stats = {
                "touch_count": len(cluster_data),
                "avg_volume": cluster_data['volume'].mean(),
                "tightness": cluster_data['close'].std() if len(cluster_data) > 1 else 0
            }
            
            # labeling logic for training phase
            label = 1 if stats['touch_count'] >= median_touches else 0
            
            features.append({
                "cluster_id": cid,
                "touch_count": stats['touch_count'],
                "avg_volume": stats['avg_volume'],
                "tightness": stats['tightness'],
                "label": label 
            })
        return pd.DataFrame(features)

    def fit(self, X, y=None):
        logger.info("[TRAIN] Starting model training...")

        # Clean and preprocess data
        df = self._validate_input(X) 
        df_clean = self._clean_data(df)       
        
        # Clustering
        df_clustered, zones = self._generate_zones(df_clean)
        
        # Extract Features
        feature_df = self._extract_features(df_clustered, zones)
            
        # 3. Train Validator
        feature_df = self._extract_features(df_clean, zones)
        
        if feature_df.empty:
            raise ValueError("No valid clusters found to train on.")

        train_X = feature_df[['touch_count', 'avg_volume', 'tightness']]
        train_y = feature_df['label']
        
        # Force variance if needed
        if len(train_y.unique()) < 2:
             logger.warning("   [Validate] Low variance in zone quality. Forcing synthetic split.")
             train_y.iloc[:len(train_y)//2] = 1
             train_y.iloc[len(train_y)//2:] = 0

        logger.info(f"   [Validate] Training Random Forest ({self.rf_trees} trees)...")     
        self.validator_model.fit(train_X, train_y)
        self.is_trained = True

        logger.info("[TRAIN] model training complete")
        return self

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model is not trained! Run .fit() first.")

        logger.info("[PREDICT] Starting inference...")

        df = self._validate_input(X)
        df_clean = self._clean_data(df)
        df_clustered, zones = self._generate_zones(df_clean)
        
        # Validate
        feature_df = self._extract_features(df_clustered, zones)
        
        logger.info("   [Inference] Scoring zones using trained Random Forest...")
        pred_X = feature_df[['touch_count', 'avg_volume', 'tightness']]
        probs = self.validator_model.predict_proba(pred_X)[:, 1]
        
        final_zones = []
        for idx, row in feature_df.iterrows():
            cid = row['cluster_id']
            score = probs[idx] * 100
            zone_info = next(z for z in zones if z['cluster_id'] == cid)
            
            status = "Insignificant"
            if score > 75: status = "Major Zone"
            elif score > 40: status = "Weak Zone"
            
            final_zones.append({
                "price_level": round(zone_info['price_level'], 2),
                "confidence": round(score, 2),
                "status": status,
                "metrics": {
                    "touches": int(row['touch_count']),
                    "volume": float(row['avg_volume'])
                }
            })
            
        return sorted(final_zones, key=lambda x: x['price_level'])
    
    def save(self, filepath):
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        logger.info(f"Loading model from {filepath}")
        return joblib.load(filepath)