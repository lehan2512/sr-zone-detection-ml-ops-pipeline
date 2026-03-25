import pandas as pd
import logging
import sys
import os

from src.sr_detection import SRZoneDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

CONFIG = {
    "raw_data": "datasets/btc_dataset.csv",
    "output_dir": "output/production_models",
    "model_artifact": "output/production_models/sr_detector_v1.pkl"
}

def run_training_pipeline():
    logger.info("=== INITIALIZING PIPELINE ===")
    
    if not os.path.exists(CONFIG["output_dir"]):
        os.makedirs(CONFIG["output_dir"])

    if not os.path.exists(CONFIG["raw_data"]):
        logger.error(f"Critical: File {CONFIG['raw_data']} not found.")
        return

    try:
        df_train = pd.read_csv(CONFIG["raw_data"])
        logger.info(f"Loaded {len(df_train)} rows from {CONFIG['raw_data']}")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return

    # Instantiate & Train
    detector = SRZoneDetector(dbscan_eps=0.05, min_samples=5)
    
    try:
        detector.fit(df_train)
    except Exception as e:
        logger.error(f"Training Failed: {e}")
        return

    detector.save(CONFIG["model_artifact"])
    logger.info("=== PIPELINE SUCCESS ===")
    logger.info(f"Ready for deployment: {CONFIG['model_artifact']}")

if __name__ == "__main__":
    run_training_pipeline()