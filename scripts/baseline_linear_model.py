import pandas as pd
import numpy as np
import json
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('models/linear_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """data loading with validation"""
    
    def __init__(self, data_dir: str = 'data/processed'):  
        self.data_dir = Path(data_dir)
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate raw data"""
        try:
            logger.info("Loading raw data...")
            possible_paths = [
                self.data_dir / 'cleaned_listings.csv',
                Path('data') / 'cleaned_listings.csv',
                Path('processed') / 'cleaned_listings.csv'
            ]
            
            for path in possible_paths:
                if path.exists():
                    df = pd.read_csv(path)
                    logger.info(f"Data loaded from {path}. Shape: {df.shape}")
                    return df
                    
            raise FileNotFoundError(
                f"Could not find cleaned_listings.csv in any of: {possible_paths}"
            )
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise



class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom feature selection and engineering"""
    
    def __init__(self, target: str = 'log_price'):
        self.target = target
        self.feature_columns = None
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and engineer features"""
        X_transformed = X.copy()
        
        # Feature selection
        potential_features = [
            'accommodates', 'bedrooms', 'bathrooms', 'review_scores_rating',
            'minimum_nights', 'host_is_superhost', 'neighbourhood', 'room_type'
        ]
        
        # Only keep existing features
        self.feature_columns = [f for f in potential_features if f in X.columns]
        X_transformed = X_transformed[self.feature_columns + [self.target]]
        
        # Convert booleans to numeric with NaN handling
        if 'host_is_superhost' in self.feature_columns:
           
            X_transformed['host_is_superhost'] = (
                X_transformed['host_is_superhost']
                .replace({'t': True, 'f': False})
                .fillna(False)
                .astype(int)
            )
            
        return X_transformed

class DataSplitter:
    """train-test splitting"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        
    def split(self, df: pd.DataFrame, target: str) -> Tuple:
        """Split into features and target"""
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )

class ModelTrainer:
    """model training and evaluation"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self._create_pipeline()
        
    def _create_pipeline(self) -> None:
        """Build preprocessing and modeling pipeline"""
        numeric_features = ['accommodates', 'bedrooms', 'bathrooms', 
                          'review_scores_rating', 'minimum_nights']
        categorical_features = ['neighbourhood', 'room_type']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('feature_selector', RFE(LinearRegression(), n_features_to_select=5)),
            ('regressor', LinearRegression())
        ])
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train model with cross-validation"""
        try:
            logger.info("Training model with cross-validation...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=5, scoring='r2'
            )
            
            # Full training
            self.model.fit(X_train, y_train)
            
            return {
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'features': self._get_feature_names()
            }
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        preds = self.model.predict(X_test)
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, preds)),
            'mae': mean_absolute_error(y_test, preds),
            'r2': r2_score(y_test, preds),
            'predictions': preds
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        try:
            # Numeric features
            num_features = self.preprocessor.named_transformers_['num'].named_steps['scaler'].get_feature_names_out()
            
            # Categorical features
            cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out()
            
            return list(num_features) + list(cat_features)
        except Exception as e:
            logger.warning(f"Could not get feature names: {str(e)}")
            return []

class ResultVisualizer:
    
    @staticmethod
    def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
        """Plot residual analysis"""
        residuals = y_true - y_pred
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(residuals, kde=True)
        plt.title('Residual Distribution')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Predicted')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig(save_path / 'residual_analysis.png')
        plt.close()

class ModelSaver:
    
    @staticmethod
    def save(model: Pipeline, metrics: Dict, save_dir: Path) -> None:
        """Save model and metrics"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, save_dir / 'linear_model.pkl')
        
        # Save metrics
        with open(save_dir / 'model_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logger.info(f"Model artifacts saved to {save_dir}")

def main():
    try:
        data_loader = DataLoader()
        feature_engineer = FeatureSelector()
        data_splitter = DataSplitter()
        model_trainer = ModelTrainer()
        visualizer = ResultVisualizer()
        model_saver = ModelSaver()
        
        # Data pipeline
        raw_data = data_loader.load_data()
        processed_data = feature_engineer.transform(raw_data)
        X_train, X_test, y_train, y_test = data_splitter.split(processed_data, 'log_price')
        
        # Model pipeline
        cv_results = model_trainer.train(X_train, y_train)
        test_results = model_trainer.evaluate(X_test, y_test)
        
        # Combine results
        full_results = {
            'cross_validation': {
                'mean_r2': cv_results['cv_mean'],
                'std_r2': cv_results['cv_std']
            },
            'test_metrics': {
                'rmse': test_results['rmse'],
                'mae': test_results['mae'],
                'r2': test_results['r2']
            },
            'selected_features': cv_results['features']
        }
        
        # Visualization and saving
        visualizer.plot_residuals(y_test, test_results['predictions'], Path('outputs'))
        model_saver.save(model_trainer.model, full_results, Path('models'))
        
        logger.info(f"""
        Model Training Complete
        =======================
        Cross-Validation R²: {cv_results['cv_mean']:.3f} ± {cv_results['cv_std']:.3f}
        Test Metrics:
          RMSE: {test_results['rmse']:.3f}
          MAE: {test_results['mae']:.3f}
          R²: {test_results['r2']:.3f}
        """)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()