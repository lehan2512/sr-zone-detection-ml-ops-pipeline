import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Tuple, List, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# Use a module-level logger
logger = logging.getLogger(__name__)

class SRModelTrainer:
    """
    Handles training, evaluation, and persistence of the S/R detection model.
    """
    def __init__(self, config: Optional[dict] = None):
        # Default hyperparameters
        self.params = {
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "features": ['volume', 'rsi', 'atr', 'vol_ma', 'rsi_lag1', 'vol_ma_lag1', 'price_roc'],
            "target": 'target'
        }
        if config:
            self.params.update(config)

        self.model = RandomForestClassifier(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            min_samples_split=self.params["min_samples_split"],
            class_weight='balanced',
            random_state=self.params["random_state"],
            n_jobs=-1
        )

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Splits data chronologically and applies Majority Class Undersampling."""
        logger.info("Preparing data for training (Chronological Split + Undersampling)...")
        
        split_idx = int(len(df) * (1 - self.params["test_size"]))
        
        # 1. Strictly split by time FIRST (prevent look-ahead bias)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        target_col = self.params["target"]
        features_col = self.params["features"]

        # 2. Undersample the Majority Class (0 = Normal) in the TRAINING set only
        support_count = len(train_df[train_df[target_col] == 1])
        resist_count = len(train_df[train_df[target_col] == 2])
        
        # Calculate target normal count (approx 2:1 ratio of normal to zones)
        target_normal_count = int((support_count + resist_count) * 2.0)

        normal_df = train_df[train_df[target_col] == 0]
        zones_df = train_df[train_df[target_col] != 0]

        # Failsafe: Ensure we don't try to sample more normal rows than exist
        target_normal_count = min(target_normal_count, len(normal_df))
        
        # Randomly sample the normal candles to shrink the majority class
        sampled_normal_df = normal_df.sample(n=target_normal_count, random_state=self.params["random_state"])
        
        # Recombine and shuffle the training set
        balanced_train_df = pd.concat([sampled_normal_df, zones_df]).sample(frac=1, random_state=self.params["random_state"])

        X_train = balanced_train_df[features_col]
        y_train = balanced_train_df[target_col]
        
        X_test = test_df[features_col]
        y_test = test_df[target_col]

        logger.info(f"Original Training set: {len(train_df)} records")
        logger.info(f"Undersampled Training set: {len(X_train)} records (Forced model attention)")
        logger.info(f"Testing set (Untouched reality): {len(X_test)} records")
        
        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Trains the Random Forest model."""
        logger.info("Training Random Forest Classifier...")
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Evaluates model performance."""
        logger.info("Evaluating model on test data...")
        predictions = self.model.predict(X_test)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, predictions, target_names=['Normal', 'Support', 'Resistance']))
        
        self._print_feature_importance()

    def _print_feature_importance(self) -> None:
        """Displays relative importance of input features."""
        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': self.params["features"],
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        print("\n--- Feature Importance ---")
        print(feature_imp_df.to_string(index=False))

    def save_model(self, output_dir: Union[str, Path] = "output/production_models", model_name: str = "sr_rf_model.pkl") -> None:
        """Persists the trained model to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_filepath = output_path / model_name
        joblib.dump(self.model, model_filepath)
        logger.info(f"Model successfully saved to {model_filepath}")