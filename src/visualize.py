import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import joblib
import os
from sklearn.metrics import confusion_matrix

class SRVisualizer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_path = os.path.join(base_dir, 'datasets', 'processed_btc_data.csv')
        self.model_path = os.path.join(base_dir, 'output', 'production_models', 'sr_rf_model.pkl')
        self.output_dir = os.path.join(base_dir, 'output', 'visualizations')
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Professional styling for academic reports
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Load assets
        self.df = pd.read_csv(self.data_path)
        self.model = joblib.load(self.model_path)
        self.features = ['volume', 'rsi', 'atr', 'vol_ma', 'rsi_lag1', 'vol_ma_lag1', 'price_roc']

    def generate_all(self):
        """Generates all charts for the assignment report."""
        print("Generating Feature Importance Chart...")
        self.plot_feature_importance()
        
        print("Generating Confusion Matrix Heatmap...")
        self.plot_confusion_matrix()
        
        print("Generating Price Chart with S/R Zones...")
        self.plot_price_zones()
        
        print(f"All visualizations saved to: {self.output_dir}")

    def plot_feature_importance(self):
        """Proves which financial metrics actually matter."""
        importances = self.model.feature_importances_
        
        imp_df = pd.DataFrame({
            'Feature': self.features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        ax = plt_sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
        
        plt.title('Random Forest Feature Importance for S/R Zone Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Relative Importance (Gini)', fontsize=12)
        plt.ylabel('Engineered Feature', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'), dpi=300)
        plt.close()

    def plot_confusion_matrix(self):
        """Visualizes model performance and the class imbalance reality."""
        # For a clean visual, we run predictions on the last 20% of the dataset
        split_idx = int(len(self.df) * 0.8)
        test_df = self.df.iloc[split_idx:]
        
        X_test = test_df[self.features]
        y_test = test_df['target']
        
        predictions = self.model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        plt_sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Support', 'Resistance'],
                    yticklabels=['Normal', 'Support', 'Resistance'])
        
        plt.title('Confusion Matrix: Test Set Predictions', fontsize=14, fontweight='bold')
        plt.ylabel('Actual Market State', fontsize=12)
        plt.xlabel('Predicted Market State', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

    def plot_price_zones(self):
        """Overlays the algorithmic targets on the actual price action."""
        # Plotting 60,000 records looks like a mess. We zoom in on the last 1500 records (approx 4 months of 2H data)
        zoom_df = self.df.tail(1500).copy()
        zoom_df.reset_index(drop=True, inplace=True)
        
        plt.figure(figsize=(14, 7))
        
        # Plot base price
        plt.plot(zoom_df.index, zoom_df['close'], color='black', alpha=0.6, label='BTC/USDT Close Price')
        
        # Overlay Support (Target = 1)
        support_points = zoom_df[zoom_df['target'] == 1]
        plt.scatter(support_points.index, support_points['close'], color='green', s=50, label='Identified Support Zone', zorder=5)
        
        # Overlay Resistance (Target = 2)
        resistance_points = zoom_df[zoom_df['target'] == 2]
        plt.scatter(resistance_points.index, resistance_points['close'], color='red', s=50, label='Identified Resistance Zone', zorder=5)
        
        plt.title('Algorithmic S/R Zone Detection (4-Month Zoom)', fontsize=16, fontweight='bold')
        plt.xlabel('Time (Intervals)', fontsize=12)
        plt.ylabel('Price (USDT)', fontsize=12)
        plt.legend(loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'price_zones.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visualizer = SRVisualizer(base_directory)
    visualizer.generate_all()