import os
import pandas as pd
import numpy as np
import json
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Dict, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('models/random_forest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.feature_names = None
        
    def load_and_preprocess(self) -> Tuple:
        logger.info("📥 Loading and preprocessing data...")
        try:
             
            possible_paths = [
                self.data_dir / "cleaned_listings.csv",  
                Path("data/processed/cleaned_listings.csv"),  
                Path("data/cleaned_listings.csv")  
            ]
            
            for path in possible_paths:
                if path.exists():
                    df = pd.read_csv(path)
                    logger.info(f"Data loaded successfully from: {path}")
                    
                    # One-hot encoding
                    df = pd.get_dummies(
                        df, 
                        columns=['neighbourhood', 'room_type', 'property_type'], 
                        drop_first=True
                    )
                    
                    # Prepare features and target
                    X = df.drop(columns=['price', 'log_price'], errors='ignore')
                    y = df['log_price']
                    self.feature_names = X.columns.tolist()
                    
                    return train_test_split(X, y, test_size=0.2, random_state=42)
            
            raise FileNotFoundError(
                f"Could not find cleaned_listings.csv in any of: {possible_paths}"
            )
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
class FeatureScaler:
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X_train)
    
    def transform(self, X_test: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X_test)
    
    def save(self, path: Path) -> None:
        joblib.dump(self.scaler, path)

class RandomForestTrainer:
    
    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_score = None
        
    def train_with_grid_search(
        self, 
        X_train: np.ndarray, 
        y_train: pd.Series
    ) -> RandomForestRegressor:
        """Train model """
        logger.info("🌲 Training Random Forest with hyperparameter tuning...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(
            rf, 
            param_grid, 
            cv=5,
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1, 
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = -grid.best_score_
        
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best CV RMSE: {self.best_score:.3f}")
        
        return self.model
    
    def evaluate(
        self, 
        X_test: np.ndarray, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        preds = self.model.predict(X_test)
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'r2': r2_score(y_test, preds),
            'predictions': preds
        }
    
    def save_model(self, path: Path) -> None:
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")

class ResultVisualizer:
    
    @staticmethod
    def plot_feature_importance(
        model: RandomForestRegressor, 
        feature_names: List[str],
        save_path: Path
    ) -> None:
        """Plot feature importance"""
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:][::-1]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x=importances[indices], 
            y=np.array(feature_names)[indices]
        )
        plt.title("Top 20 Feature Importances - Random Forest")
        plt.xlabel("Importance")
        plt.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")

class RandomForestPipeline:
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.scaler = FeatureScaler()
        self.trainer = RandomForestTrainer()
        self.visualizer = ResultVisualizer()
        
    def run(self):
        try:
            (X_train, X_test, y_train, y_test) = self.data_loader.load_and_preprocess()
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.trainer.train_with_grid_search(X_train_scaled, y_train)
            
            results = self.trainer.evaluate(X_test_scaled, y_test)
            
            self._save_artifacts(results)
            
            self.visualizer.plot_feature_importance(
                self.trainer.model,
                self.data_loader.feature_names,
                Path("outputs/rf_feature_importance.png")
            )
            
            logger.info(f"""
            ✅ Random Forest Training Complete
            ================================
            Best Params: {self.trainer.best_params}
            CV Best RMSE: {self.trainer.best_score:.3f}
            Test RMSE: {results['rmse']:.3f}
            Test R²: {results['r2']:.3f}
            """)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _save_artifacts(self, results: Dict) -> None:
        """Save all pipeline artifacts"""
        # Create directories if they don't exist
        Path("models").mkdir(exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)
        
        self.trainer.save_model(Path("models/random_forest_model.pkl"))
        self.scaler.save(Path("models/rf_scaler.pkl"))
        
        with open(Path("outputs/rf_metrics.json"), "w") as f:
            json.dump({
                'best_params': self.trainer.best_params,
                'cv_rmse': self.trainer.best_score,
                'test_rmse': results['rmse'],
                'test_r2': results['r2']
            }, f, indent=4)

if __name__ == "__main__":
    pipeline = RandomForestPipeline()
    pipeline.run()