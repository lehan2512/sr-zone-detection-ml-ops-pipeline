import sys
import pandas as pd
import json
import logging
from src.sr_detection import SRZoneDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger()

def run_inference(target_csv, model_path='output/production_models/sr_detector_v1.pkl'):
    try:
        logger.info(f"Loading model from {model_path}...")
        detector = SRZoneDetector.load(model_path)
    except FileNotFoundError:
        logger.error("Model artifact not found. Did you run pipeline.py?")
        return

    try:
        logger.info(f"Loading target data: {target_csv}...")
        df_target = pd.read_csv(target_csv)
    except FileNotFoundError:
        logger.error(f"Target file {target_csv} not found.")
        return

    try:
        zones = detector.predict(df_target)
        
        print("\n=== INFERENCE RESULTS ===")
        print(json.dumps(zones, indent=4))
        
        with open('output/inference_output.json', 'w') as f:
            json.dump(zones, f, indent=4)
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")

if __name__ == "__main__":
    run_inference('datasets/eth_dataset.csv')