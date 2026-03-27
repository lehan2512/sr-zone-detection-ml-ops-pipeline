import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import joblib
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix

class SRVisualizer:
    def __init__(self, data_path: str, model_path: str, output_dir: str, symbol: str):
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.symbol = symbol
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Professional styling for reports and UI
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Load assets
        self.df = pd.read_csv(self.data_path)
        self.model = joblib.load(self.model_path)
        self.features = ['volume', 'rsi', 'atr', 'vol_ma', 'rsi_lag1', 'vol_ma_lag1', 'price_roc']

    def generate_all(self) -> dict:
        """Generates all charts and returns their file paths."""
        paths = {}
        paths['feature_importance'] = self.plot_feature_importance()
        paths['confusion_matrix'] = self.plot_confusion_matrix()
        paths['price_zones'] = self.plot_price_zones()
        return paths

    def plot_feature_importance(self) -> str:
        importances = self.model.feature_importances_
        imp_df = pd.DataFrame({'Feature': self.features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        ax = plt_sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
        plt.title(f'Feature Importance: {self.symbol}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'feature_importance.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        return str(save_path)

    def plot_confusion_matrix(self) -> str:
        split_idx = int(len(self.df) * 0.8)
        test_df = self.df.iloc[split_idx:]
        
        X_test = test_df[self.features]
        y_test = test_df['target']
        
        predictions = self.model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        plt_sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Support', 'Resistance'], yticklabels=['Normal', 'Support', 'Resistance'])
        plt.title(f'Confusion Matrix: {self.symbol}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'confusion_matrix.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        return str(save_path)

    def plot_price_zones(self) -> str:
        zoom_df = self.df.tail(1500).copy()
        zoom_df.reset_index(drop=True, inplace=True)
        
        plt.figure(figsize=(14, 7))
        plt.plot(zoom_df.index, zoom_df['close'], color='black', alpha=0.6, label=f'{self.symbol} Close Price')
        
        support_points = zoom_df[zoom_df['target'] == 1]
        plt.scatter(support_points.index, support_points['close'], color='green', s=50, label='Identified Support Zone', zorder=5)
        
        resistance_points = zoom_df[zoom_df['target'] == 2]
        plt.scatter(resistance_points.index, resistance_points['close'], color='red', s=50, label='Identified Resistance Zone', zorder=5)
        
        plt.title(f'Algorithmic S/R Zone Detection ({self.symbol})', fontsize=16, fontweight='bold')
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        save_path = self.output_dir / f'price_zones.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        return str(save_path)