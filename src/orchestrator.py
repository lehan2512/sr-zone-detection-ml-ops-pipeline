import json
import logging
import sys
import yaml
import argparse
import pandas as pd
from pathlib import Path

# Import your pipeline modules
from fetch import fetch_binance_klines
from data_processor import SREngineer
from train import SRModelTrainer
from visualize import SRVisualizer

# Use a module-level logger
logger = logging.getLogger(__name__)

def setup_logging(symbol: str, base_dir: Path):
    """Configures standard structured logging for the pipeline run."""
    log_dir = base_dir / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{symbol}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True # Forces a reset of the logging config for sequential runs
    )

def load_config(config_path: Path) -> dict:
    """Loads the base YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_ml_microservice(symbol: str) -> str:
    """
    The main orchestrator function. 
    Returns a JSON string containing the S/R zones and chart paths for the UI.
    """
    # 1. Setup Environment Paths
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "config.yaml"
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    
    # Initialize standard logging
    setup_logging(symbol, base_dir)
    logger.info(f"--- INITIALIZING ML-OPS PIPELINE FOR: {symbol} ---")

    # 2. Dynamic File Paths for this specific symbol
    raw_data_path = datasets_dir / f"{symbol}_raw.csv"
    processed_data_path = datasets_dir / f"{symbol}_processed.csv"

    try:
        # 3. Load Base Config & Apply Dynamic Overrides
        config = load_config(config_path)
        config['data']['raw_path'] = str(raw_data_path)
        config['data']['processed_path'] = str(processed_data_path)

        # --- PHASE 1: Data Ingestion ---
        logger.info(f"PHASE 1: Fetching latest 60,000 hourly candles for {symbol}...")
        fetch_binance_klines(symbol=symbol, interval='1h', total_records=60000, filename=str(raw_data_path))

        # --- PHASE 2: Data Processing & Feature Engineering ---
        logger.info("PHASE 2: Initializing Feature Engineering...")
        engineer = SREngineer(config=config.get("features", {}))
        processed_df = engineer.process_pipeline(raw_data_path)

        # --- PHASE 3: Machine Learning ---
        logger.info("PHASE 3: Splitting data and generating targets (Fixed Leakage)...")
        trainer = SRModelTrainer(config=config.get("train", {}))
        
        # 3.1 Initial Chronological Split
        test_size = config.get("train", {}).get("test_size", 0.2)
        split_idx = int(len(processed_df) * (1 - test_size))
        
        train_df_raw = processed_df.iloc[:split_idx].copy()
        test_df_raw = processed_df.iloc[split_idx:].copy()

        # 3.2 Fit Clusters on TRAINING data only
        engineer.fit_clusters(train_df_raw)
        
        # 3.3 Label targets for both sets using training clusters
        train_df_labeled = engineer.label_targets(train_df_raw)
        test_df_labeled = engineer.label_targets(test_df_raw)

        # Combine for saving/visualization purposes (optional but keeps existing flow)
        full_processed_df = pd.concat([train_df_labeled, test_df_labeled])
        full_processed_df.to_csv(processed_data_path, index=False)

        # 3.4 Prepare balanced training set and final features
        logger.info("PHASE 3.4: Initializing Random Forest Hyperparameter Tuning & Training...")
        X_train, X_test, y_train, y_test = trainer.prepare_data_from_split(train_df_labeled, test_df_labeled)
        
        trainer.train(X_train, y_train)

        model_path = base_dir / "output" / "production_models" / "sr_rf_model.pkl"
        trainer.save_model(output_dir=model_path.parent, model_name=model_path.name)

        # --- PHASE 4: Visual Generation ---
        logger.info(f"PHASE 4: Generating charts for UI render ({symbol})...")
        viz_dir = base_dir / "output" / "visualizations"
        visualizer = SRVisualizer(
            data_path=str(processed_data_path),
            model_path=str(model_path),
            output_dir=str(viz_dir),
            symbol=symbol
        )
        chart_paths = visualizer.generate_all()

        # --- PHASE 5: UI JSON Payload Generation ---
        logger.info(f"PHASE 5: Packaging {symbol} S/R zones and chart paths for UI payload...")
        payload = {
            "symbol": symbol,
            "status": "success",
            "data": {
                "support_zones": [round(val, 2) for val in engineer.support_centers],
                "resistance_zones": [round(val, 2) for val in engineer.resistance_centers]
            },
            "charts": {
                "price_zones_url": chart_paths['price_zones'],
                "feature_importance_url": chart_paths['feature_importance'],
                "confusion_matrix_url": chart_paths['confusion_matrix']
            },
            "message": "Pipeline executed successfully."
        }

        json_output = json.dumps(payload, indent=4)
        logger.info(f"--- PIPELINE COMPLETE FOR {symbol} ---")
        return json_output

    except Exception as e:
        # Standard error handling with exc_info=True to print the full stack trace to the log
        logger.error(f"Pipeline failed for {symbol}: {str(e)}", exc_info=True)
        error_payload = {
            "symbol": symbol,
            "status": "error",
            "message": str(e)
        }
        return json.dumps(error_payload, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SR Detection ML Pipeline.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="The Binance trading pair symbol (e.g., ETHUSDT)")
    args = parser.parse_args()

    final_json = run_ml_microservice(args.symbol)
    
    print("\n" + "="*50)
    print("FINAL UI JSON PAYLOAD:")
    print("="*50)
    print(final_json)