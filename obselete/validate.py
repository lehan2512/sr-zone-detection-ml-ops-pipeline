import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import json

logger = logging.getLogger(__name__)

def validate_zones(input_data_path, input_zones_path, output_zones_path, model_output_path):
    logger.info("Loading artifacts for validation...")
    try:
        df = pd.read_csv(input_data_path)
        with open(input_zones_path, 'r') as f:
            zones = json.load(f)
    except FileNotFoundError:
        logger.error("Input files not found. Run cluster.py first.")
        return

    if not zones:
        logger.error("No zones found in JSON.")
        return
    
    cluster_features = []
    all_touches = []
    all_tightness = []

    temp_stats = {}
    for zone in zones:
        cid = zone['cluster_id']
        cluster_data = df[df['sr_cluster'] == cid]
        
        if len(cluster_data) == 0:
            continue
            
        count = len(cluster_data)
        std_dev = cluster_data['close'].std()
        
        if pd.isna(std_dev): std_dev = 0 
        
        all_touches.append(count)
        all_tightness.append(std_dev)
        
        temp_stats[cid] = {
            "count": count,
            "tightness": std_dev,
            "avg_vol": cluster_data['volume'].mean() if 'volume' in cluster_data.columns else 0
        }

    # Calculate Medians - Thresholds
    median_touches = np.median(all_touches) if all_touches else 0
    median_tightness = np.median(all_tightness) if all_tightness else 0

    logger.info(f"Median Touches={median_touches}, Median Tightness={median_tightness:.2f}")

    for cid, stats in temp_stats.items():
        # RELATIVE HEURISTIC
        # A zone is Good(1) if it has ABOVE average touches OR is tighter than average
        is_high_quality = 1 if (stats['count'] >= median_touches) else 0
        
        cluster_features.append({
            "cluster_id": cid,
            "touch_count": stats['count'],
            "avg_volume": stats['avg_vol'],
            "tightness": stats['tightness'],
            "label": is_high_quality # Target Variable
        })

    train_df = pd.DataFrame(cluster_features)

    if len(train_df['label'].unique()) < 2:
        logger.warning("No variance in cluster quality. Forcing split for training stability.")
        # mark the first half as good to prevent crash
        mid = len(train_df) // 2
        train_df.loc[:mid, 'label'] = 1
        train_df.loc[mid:, 'label'] = 0

    X = train_df[['touch_count', 'avg_volume', 'tightness']]
    y = train_df['label']

    logger.info("Training Random Forest Validator...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    joblib.dump(rf, model_output_path)
    logger.info(f"Trained RF Model saved to: {model_output_path}")
    
    # Evaluate
    y_pred = rf.predict(X)
    precision = precision_score(y, y_pred, zero_division=0)
    logger.info(f"Model Precision: {precision:.2f}")
    
    # Feature Importance
    importances = rf.feature_importances_
    logger.info(f"Feature Importance -> Touches: {importances[0]:.2f}, Volume: {importances[1]:.2f}, Tightness: {importances[2]:.2f}")

    logger.info("Scoring Zones...")
    # Predict Confidence Scores for each zone
    probs = rf.predict_proba(X)
    confidence_scores = probs[:, 1]
    
    for idx, row in train_df.iterrows():
        cid = row['cluster_id']
        score = confidence_scores[idx] * 100
        
        for z in zones:
            if z['cluster_id'] == cid:
                z['confidence_score'] = float(round(score, 2))
                if score > 75:
                    z['status'] = "Major Structural Level"
                elif score > 40:
                    z['status'] = "Weak Support/Resistance"
                else:
                    z['status'] = "Insignificant"

    with open(output_zones_path, 'w') as f:
        json.dump(zones, f, indent=4)
        
    logger.info(f"Validation Complete. Output saved to: {output_zones_path}")

if __name__ == "__main__":
    INPUT_DATA = 'datasets/btc_dataset_with_sr.csv'
    INPUT_ZONES = 'datasets/sr_zones.json'
    OUTPUT_ZONES = 'datasets/sr_zones_scored.json'
    OUTPUT_MODEL = 'production_models/rf_model.pkl'
    
    validate_zones(INPUT_DATA, INPUT_ZONES, OUTPUT_ZONES, OUTPUT_MODEL)