import os
import sys
import logging

try:
    from clean import clean_btc_data
    from cluster import generate_sr_zones
    from validate import validate_zones
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules. {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "raw_data": "datasets/btc_dataset.csv",
    
    "working_dir": "datasets",
    "deploy_dir": "production_models",
    
    "clean_data": "datasets/btc_dataset_cleaned.csv",
    "labeled_data": "datasets/btc_dataset_with_sr.csv",
    "zones_temp": "datasets/sr_zones_temp.json", 
    
    "kmeans_model": "production_models/kmeans_model.pkl",
    "rf_model": "production_models/rf_model.pkl",
    "final_artifact": "production_models/sr_zones_scored.json"
}

def run_pipeline():
    logger.info("--- STARTING S/R DETECTION PIPELINE ---")

    for directory in [CONFIG["working_dir"], CONFIG["deploy_dir"]]:
        if not os.path.exists(directory):
            logger.info(f"Creating directory: {directory}")
            os.makedirs(directory)

    # Data Cleaning
    if not os.path.exists(CONFIG["raw_data"]):
        logger.error(f"Raw file {CONFIG['raw_data']} not found!")
        return
    
    clean_btc_data(CONFIG["raw_data"])
    
    if not os.path.exists(CONFIG["clean_data"]):
        logger.error("Cleaning failed. Pipeline Aborted.")
        return

    # Clustering
    generate_sr_zones(
        CONFIG["clean_data"], 
        CONFIG["labeled_data"], 
        CONFIG["zones_temp"],
        CONFIG["kmeans_model"]
    )

    if not os.path.exists(CONFIG["zones_temp"]):
        logger.error("Clustering failed. Pipeline Aborted.")
        return

    # Validation
    validate_zones(
        CONFIG["labeled_data"], 
        CONFIG["zones_temp"], 
        CONFIG["final_artifact"],
        CONFIG["rf_model"]
    )

    # Verification
    if os.path.exists(CONFIG["final_artifact"]):
        logger.info(f"--- PIPELINE SUCCESS ---")
        logger.info(f"Final Deployment Artifact saved to: {CONFIG['final_artifact']}")
    else:
        logger.error("Validation failed.")

if __name__ == "__main__":
    run_pipeline()