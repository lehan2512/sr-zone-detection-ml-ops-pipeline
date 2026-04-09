import json
import logging
import sys
import yaml
import argparse
import pandas as pd
import warnings
from pathlib import Path

# Suppress annoying sklearn warnings that clutter the output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Import your pipeline modules
from fetch import fetch_binance_klines
from data_processor import SREngineer
from train import SRModelTrainer
from visualize import SRVisualizer
from blob_manager import upload_to_blob, download_blob

# Use a module-level logger
logger = logging.getLogger(__name__)

def setup_logging(symbol: str, base_dir: Path, quiet: bool = False):
    """Configures standard structured logging for the pipeline run."""
    log_dir = base_dir / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{symbol}.log"

    handlers = [logging.FileHandler(log_file)]
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True 
    )

def load_config(config_path: Path) -> dict:
    """Loads the base YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Base config not found at {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_ml_microservice(symbol: str, mode: str = "train", quiet: bool = False) -> str:
    """
    The main orchestrator function. 
    Supports modes: train, infer, verify
    """
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "config.yaml"
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    output_dir = base_dir / "output" / "production_models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(symbol, base_dir, quiet=quiet)
    logger.info(f"--- INITIALIZING ML-OPS PIPELINE | SYMBOL: {symbol} | MODE: {mode.upper()} ---")

    raw_data_path = datasets_dir / f"{symbol}_raw.csv"
    processed_data_path = datasets_dir / f"{symbol}_processed.csv"
    champion_model_path = output_dir / f"champion_model_{symbol}.pkl"
    challenger_model_path = output_dir / f"challenger_model_{symbol}.pkl"

    try:
        config = load_config(config_path)
        engineer = SREngineer(config=config.get("features", {}))
        trainer = SRModelTrainer(config=config.get("train", {}))

        # --- MODE: VERIFY (CI/CD Check) ---
        if mode == "verify":
            logger.info("PHASE: Verification Mode. Fetching minimal data...")
            fetch_binance_klines(symbol=symbol, interval='1h', total_records=100, filename=str(raw_data_path))
            engineer.process_pipeline(raw_data_path)
            logger.info("Verification successful.")
            return json.dumps({"status": "verified", "symbol": symbol})

        # --- MODE: INFER (On-Demand Inference) ---
        if mode == "infer":
            logger.info("PHASE: Inference Mode. Fetching latest data...")
            fetch_binance_klines(symbol=symbol, interval='1h', total_records=60000, filename=str(raw_data_path))
            
            # Download Champion from Blob Storage
            success = download_blob(container_name="models", blob_name=f"sr_rf_model_{symbol}.pkl", download_path=champion_model_path)
            if not success:
                raise Exception(f"Could not find a champion model for {symbol} in storage. Run training first.")

            trainer.load_model(champion_model_path)
            processed_df = engineer.process_pipeline(raw_data_path)
            
            # For inference, we still need to fit clusters to get support/resistance center
            engineer.fit_clusters(processed_df)
            
            payload = {
                "symbol": symbol,
                "status": "success",
                "mode": "inference",
                "data": {
                    "support_zones": sorted([round(val, 2) for val in engineer.support_centers]),
                    "resistance_zones": sorted([round(val, 2) for val in engineer.resistance_centers], reverse=True)
                }
            }
            return json.dumps(payload, indent=4)

        # --- MODE: TRAIN (Scheduled Champion/Challenger) ---
        if mode == "train":
            logger.info("PHASE: Training Mode. Fetching 60,000 candles...")
            fetch_binance_klines(symbol=symbol, interval='1h', total_records=60000, filename=str(raw_data_path))
            
            processed_df = engineer.process_pipeline(raw_data_path)
            test_size = config.get("train", {}).get("test_size", 0.2)
            split_idx = int(len(processed_df) * (1 - test_size))
            
            train_df_raw = processed_df.iloc[:split_idx].copy()
            test_df_raw = processed_df.iloc[split_idx:].copy()

            engineer.fit_clusters(train_df_raw)
            train_df_labeled = engineer.label_targets(train_df_raw)
            test_df_labeled = engineer.label_targets(test_df_raw)

            X_train, X_test, y_train, y_test = trainer.prepare_data_from_split(train_df_labeled, test_df_labeled)
            
            logger.info("Training Challenger model...")
            trainer.train(X_train, y_train)
            trainer.save_model(output_dir=output_dir, model_name=f"challenger_model_{symbol}.pkl")

            # Download existing Champion to compare
            download_blob(container_name="models", blob_name=f"sr_rf_model_{symbol}.pkl", download_path=champion_model_path)
            
            is_better = trainer.evaluate_champion_vs_challenger(X_test, y_test, champion_model_path)

            if is_better:
                logger.info("New model is better! Promoting Challenger to Champion.")
                trainer.save_model(output_dir=output_dir, model_name=f"sr_rf_model_{symbol}.pkl")
                upload_to_blob(output_dir / f"sr_rf_model_{symbol}.pkl", container_name="models")
                message = "New model promoted to Champion."
            else:
                logger.info("Challenger failed to beat Champion. Keeping existing model.")
                message = "Champion retained."

            return json.dumps({"status": "success", "symbol": symbol, "message": message, "challenger_won": is_better})

    except Exception as e:
        logger.error(f"Pipeline failed for {symbol}: {str(e)}", exc_info=True)
        return json.dumps({"symbol": symbol, "status": "error", "message": str(e)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SR Detection ML Orchestrator.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--mode", type=str, choices=["train", "infer", "verify"], default="train")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    final_json = run_ml_microservice(args.symbol, mode=args.mode, quiet=args.quiet)
    print(final_json)
