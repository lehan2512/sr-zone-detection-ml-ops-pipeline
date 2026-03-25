import os
from data_processor import SREngineer
from train import SRModelTrainer  # Add this import

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'datasets', 'btc_dataset.csv')

    print(f"Looking for dataset at: {dataset_path}")

    # --- PHASE 1: Data Processing ---
    # We now pass min_clusters and max_clusters instead of n_clusters
    engineer = SREngineer(
        rsi_period=14, 
        atr_period=14, 
        vol_ma_period=20, 
        extrema_window=20, 
        min_clusters=2, 
        max_clusters=8
    )
    processed_df = engineer.process_pipeline(dataset_path)
    
    # Save the processed data for our visualization script
    processed_df.to_csv(os.path.join(base_dir, 'datasets', 'processed_btc_data.csv'), index=False)

    # --- PHASE 2: Model Training & Evaluation ---
    trainer = SRModelTrainer(test_size=0.2)
    
    # Split chronologically
    X_train, X_test, y_train, y_test = trainer.prepare_data(processed_df)
    
    # Train
    trainer.train(X_train, y_train)
    
    # Evaluate
    trainer.evaluate(X_test, y_test)
    
    # Save Model
    model_dir = os.path.join(base_dir, 'output', 'production_models')
    trainer.save_model(output_dir=model_dir)

if __name__ == "__main__":
    main()