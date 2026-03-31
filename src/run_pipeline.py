import yaml
import logging
import sys
from pathlib import Path
from data_processor import SREngineer
from train import SRModelTrainer

def setup_logging():
    """Configures global logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log")
        ]
    )

def load_config(config_path: Path) -> dict:
    """Loads YAML configuration file."""
    if not config_path.exists():
        logging.warning(f"Config not found at {config_path}. Using default parameters.")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 1. Setup Environment
    setup_logging()
    logger = logging.getLogger("pipeline")
    
    # Use pathlib for robust path management
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / "config" / "config.yaml"
    
    # 2. Load Configuration
    config = load_config(config_path)
    
    # Extract sub-configs
    data_cfg = config.get("data", {})
    feat_cfg = config.get("features", {})
    train_cfg = config.get("train", {})
    
    # 3. Initialize Components
    # Pass the feature config to the engineer
    engineer = SREngineer(config=feat_cfg)
    
    # Pass the training config to the trainer
    trainer = SRModelTrainer(config=train_cfg)

    # 4. Execute Pipeline
    try:
        raw_data_path = base_dir / data_cfg.get("raw_path", "datasets/btc_dataset.csv")
        processed_data_path = base_dir / data_cfg.get("processed_path", "datasets/processed_btc_data.csv")
        
        logger.info(f"--- PHASE 1: Data Processing ({raw_data_path.name}) ---")
        processed_df = engineer.process_pipeline(raw_data_path)
        
        logger.info("--- PHASE 2: Target Labeling (Preventing Leakage) ---")
        # Split for clustering fit (using the same split ratio as trainer)
        split_idx = int(len(processed_df) * (1 - train_cfg.get("test_size", 0.2)))
        train_df_for_fit = processed_df.iloc[:split_idx].copy()
        
        # Fit clusters on training data only
        engineer.fit_clusters(train_df_for_fit)
        
        # Label the entire dataset based on these clusters
        labeled_df = engineer.label_targets(processed_df)
        
        # Save processed data
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        labeled_df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed and labeled data saved to {processed_data_path}")

        logger.info("--- PHASE 3: Model Training & Evaluation ---")
        X_train, X_test, y_train, y_test = trainer.prepare_data(labeled_df)
        
        trainer.train(X_train, y_train)
        trainer.evaluate(X_test, y_test)
        
        # 5. Save Artifacts
        model_dir = base_dir / train_cfg.get("output_dir", "output/production_models")
        model_name = train_cfg.get("model_name", "sr_rf_model.pkl")
        trainer.save_model(output_dir=model_dir, model_name=model_name)
        
        logger.info("Pipeline Execution Successful.")

    except Exception as e:
        logger.error(f"Pipeline Execution Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()