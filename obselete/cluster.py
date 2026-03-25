import logging
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)
    
# Elbow Method to find optimal K
def find_optimal_k(data, max_k=10):
    logger.info("Tuning Hyperparameters (Elbow Method & Inertia)...")

    inertias = []
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    x_points = range(2, max_k + 1)
    
    # Vector math to find the point furthest from the straight line
    p1 = np.array([x_points[0], inertias[0]])
    p2 = np.array([x_points[-1], inertias[-1]])

    best_k = 2
    max_dist = 0
    
    for i, k in enumerate(x_points):
        p = np.array([k, inertias[i]])
        # Calculate perpendicular distance from point p to line p1-p2
        dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
        
        if dist > max_dist:
            max_dist = dist
            best_k = k

    logger.info(f"Optimization Complete. Elbow detected at k={best_k}")
    return best_k

def generate_sr_zones(input_file_path, output_file_path_data, output_file_path_zones, model_output_path):
    logger.info("Loading Cleaned Data...")
    try:
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        logger.error(f"File not found: '{input_file_path}'")
        return None

    X = df[['close']].values

    # Dynamic K Selection based on density.
    optimal_k = find_optimal_k(X)

    # K-Means clustering
    logger.info(f"Applying K-Means with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['sr_cluster'] = kmeans.fit_predict(X)
    
    joblib.dump(kmeans, model_output_path)
    logger.info(f"Trained K-Means Model saved to: {model_output_path}")

    centroids = kmeans.cluster_centers_
    
    zones_list = []
    for i, center in enumerate(centroids):
        zones_list.append({
            "cluster_id": int(i),
            "price_level": float(center[0]), # The S/R Price
            "member_count": int(len(df[df['sr_cluster'] == i])) # How many times price was here
        })

    # Sort zones by price level    
    zones_list.sort(key=lambda x: x['price_level'])

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file_path_data), exist_ok=True)
    os.makedirs(os.path.dirname(output_file_path_zones), exist_ok=True)
    
    # Artifact A: The dataset with Cluster IDs (Used for RF Training)
    df.to_csv(output_file_path_data, index=False)
    
    # Artifact B: The Zones
    with open(output_file_path_zones, 'w') as f:
        json.dump(zones_list, f, indent=4)

    logger.info(f"Clustering artifacts saved.")
    logger.info(f"Labeled Data: {output_file_path_data}")
    logger.info(f"Candidate Zones: {output_file_path_zones}")
    logger.info(f" /nIdentified {optimal_k} significant levels.")

if __name__ == "__main__":
    INPUT_FILE = 'datasets/btc_dataset_cleaned.csv'
    OUTPUT_DATA = 'datasets/btc_dataset_with_sr.csv'
    OUTPUT_ZONES = 'datasets/sr_zones.json'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    generate_sr_zones('datasets/btc_dataset_cleaned.csv', 
                      'datasets/btc_dataset_with_sr.csv', 
                      'datasets/sr_zones_temp.json', 
                      'production_models/kmeans_model.pkl')