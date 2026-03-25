import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelTrainer')

class SRModelTrainer:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.features = ['volume', 'rsi', 'atr', 'vol_ma', 'rsi_lag1', 'vol_ma_lag1', 'price_roc']
        self.target = 'target'
        
        self.model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,          # Give it a bit more room to learn complex patterns
            min_samples_split=5, 
            class_weight='balanced', # Let sklearn handle the math on the undersampled data
            random_state=self.random_state,
            n_jobs=-1 
        )

    def prepare_data(self, df: pd.DataFrame):
        """Splits data chronologically and applies Majority Class Undersampling."""
        logger.info("Preparing data for training (Chronological Split + Undersampling)...")
        
        # Calculate split index
        split_idx = int(len(df) * (1 - self.test_size))

        # 1. Strictly split by time FIRST (prevent look-ahead bias)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # 2. Undersample the Majority Class (0 = Normal) in the TRAINING set only
        # We want the ratio of Normal to Zones to be about 2:1 instead of 10:1
        support_count = len(train_df[train_df[self.target] == 1])
        resist_count = len(train_df[train_df[self.target] == 2])
        target_normal_count = int((support_count + resist_count) * 2.0)

        normal_df = train_df[train_df[self.target] == 0]
        zones_df = train_df[train_df[self.target] != 0]

        # Randomly sample the normal candles to shrink the majority class
        sampled_normal_df = normal_df.sample(n=target_normal_count, random_state=self.random_state)
        
        # Recombine and shuffle the training set
        balanced_train_df = pd.concat([sampled_normal_df, zones_df]).sample(frac=1, random_state=self.random_state)

        X_train = balanced_train_df[self.features]
        y_train = balanced_train_df[self.target]
        
        X_test = test_df[self.features]
        y_test = test_df[self.target]

        logger.info(f"Original Training set: {len(train_df)} records")
        logger.info(f"Undersampled Training set: {len(X_train)} records (Forced model attention)")
        logger.info(f"Testing set (Untouched reality): {len(X_test)} records")
        
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Trains the Random Forest model."""
        logger.info("Training Random Forest Classifier with balanced class weights...")
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

    def evaluate(self, X_test, y_test):
        """Evaluates the model using precision, recall, and confusion matrix."""
        logger.info("Evaluating model on unseen test data...")
        predictions = self.model.predict(X_test)
        
        print("\n--- Classification Report ---")
        # Target names: 0 = Normal, 1 = Support, 2 = Resistance
        print(classification_report(y_test, predictions, target_names=['Normal', 'Support', 'Resistance']))
        
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test, predictions))
        
        self._print_feature_importance()

    def _print_feature_importance(self):
        """Extracts and prints the importance of each feature."""
        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\n--- Feature Importance ---")
        print(feature_imp_df.to_string(index=False))

    def save_model(self, output_dir="output/production_models"):
        """Saves the trained model to disk."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        model_path = os.path.join(output_dir, "sr_rf_model.pkl")
        joblib.dump(self.model, model_path)
        logger.info(f"Model successfully saved to {model_path}")