import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import os
import joblib
import logging
from typing import Tuple, Union, List, Optional
from pathlib import Path
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

class DataLoader:
    """loading data"""
    
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        
    def load_data(self) -> pd.DataFrame:
        """Load data from file"""
        logger.info(f"📂 Loading data from: {self.filepath}")
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found at {self.filepath}")
            
        try:
            return pd.read_csv(self.filepath)
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

class DataCleaner(BaseEstimator, TransformerMixin):
    """all data cleaning"""
    
    def __init__(self, 
                 drop_cols: Optional[List[str]] = None,
                 price_quantile: float = 0.99,
                 log_target: bool = True):
        """
        Initialize the DataCleaner
        """
        self.drop_cols = drop_cols or [
            'id', 'name', 'description', 'amenities', 'host_name', 'host_id',
            'host_since', 'first_review', 'last_review', 'calendar_last_scraped',
            'latitude', 'longitude', 'bathrooms_text'
        ]
        self.price_quantile = price_quantile
        self.log_target = log_target
        self.price_cap_ = None
        self.shape_before_ = None
        self.shape_after_ = None
        
    def fit(self, df: pd.DataFrame, y=None):
        """Learn parameters from data"""
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformations"""
        logger.info("🧹 Cleaning data...")
        self.shape_before_ = df.shape
        
        # Make a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()
        
        # Drop specified columns
        df_clean = self._drop_columns(df_clean)
        
        # Process percentage columns
        df_clean = self._process_percentage_columns(df_clean)
        
        # Process boolean columns
        df_clean = self._process_boolean_columns(df_clean)
        
        # Process bathrooms
        df_clean = self._process_bathrooms(df_clean)
        
        # Process price
        df_clean = self._process_price(df_clean)
        
        # Drop duplicates
        df_clean = self._drop_duplicates(df_clean)
        
        # Drop rows with missing values in key columns
        df_clean = self._drop_missing_values(df_clean)
        
        # Add log price if needed
        if self.log_target:
            df_clean['log_price'] = np.log1p(df_clean['price'])
        
        self.shape_after_ = df_clean.shape
        logger.info(f"✅ Cleaned dataset shape: {self.shape_before_} -> {self.shape_after_}")
        
        return df_clean
    
    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop specified columns"""
        cols_to_drop = [col for col in self.drop_cols if col in df.columns]
        logger.info(f"Dropping columns: {cols_to_drop}")
        return df.drop(columns=cols_to_drop, errors='ignore')
    
    def _process_percentage_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process percentage columns """
        for col in ['host_response_rate', 'host_acceptance_rate']:
            if col in df.columns:
                df[col] = df[col].str.rstrip('%').astype(float) / 100
        return df
    
    def _process_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process boolean columns"""
        if 'host_is_superhost' in df.columns:
            df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0})
        return df
    
    def _process_bathrooms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process bathrooms text"""
        if 'bathrooms_text' in df.columns:
            df['bathrooms'] = df['bathrooms_text'].str.extract(r'(\d+(\.\d+)?)')[0].astype(float)
        return df
    
    def _process_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process price column"""
        if 'price' in df.columns:
            # Remove $ and commas, convert to float
            df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
            
            df = df[df['price'] > 0]
            
            # Cap prices at specified quantile
            self.price_cap_ = df['price'].quantile(self.price_quantile)
            df = df[df['price'] <= self.price_cap_]
        return df
    
    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate rows"""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Found and dropping {duplicates} duplicate rows")
            return df.drop_duplicates()
        return df
    
    def _drop_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with missing values in key columns"""
        key_cols = ['price', 'room_type', 'neighbourhood']
        missing_before = df[key_cols].isnull().sum().sum()
        df_clean = df.dropna(subset=key_cols)
        missing_after = df_clean[key_cols].isnull().sum().sum()
        logger.info(f"Dropped {missing_before - missing_after} rows with missing values")
        return df_clean

class FeatureEncoder(BaseEstimator, TransformerMixin):
    """feature encoding and transformation"""
    
    def __init__(self, categorical_cols: Optional[List[str]] = None):
        """
        Initialize the FeatureEncoder
        
        """
        self.categorical_cols = categorical_cols or [
            'neighbourhood', 'room_type', 'property_type'
        ]
        self.encoded_columns_ = None
        
    def fit(self, df: pd.DataFrame, y=None):
        """Learn parameters from data"""
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("🔤 Encoding categorical features...")
        
        df_encoded = df.copy()
        
       
        cols_to_encode = [col for col in self.categorical_cols if col in df_encoded.columns]
        
        if cols_to_encode:
            df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True)
            self.encoded_columns_ = [col for col in df_encoded.columns if col not in df.columns]
            logger.info(f"Added {len(self.encoded_columns_)} new columns through one-hot encoding")
        else:
            logger.warning("No categorical columns found for encoding")
            
        return df_encoded

class DataSplitter:
    
    def __init__(self, 
                 test_size: float = 0.2, 
                 random_state: int = 42,
                 scale_features: bool = True):
        """
        Initialize the DataSplitter
        
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.feature_columns_ = None
        
    def split_features_target(self, df: pd.DataFrame, target_col: str = 'log_price') -> Tuple[pd.DataFrame, pd.Series]:
        """Split features and target"""
        logger.info("🧪 Splitting features and target...")
        X = df.drop(columns=[target_col, 'price'], errors='ignore')
        X = X.select_dtypes(include=[np.number])
        y = df[target_col]
        self.feature_columns_ = X.columns.tolist()
        return X, y
    
    def train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into train and test sets"""
        logger.info("🔀 Splitting train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        if self.scale_features:
            logger.info("📏 Scaling features...")
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test

class DataPipeline:
    """Main pipeline class """
    
    def __init__(self, 
                 input_path: Union[str, Path],
                 output_dir: Union[str, Path] = "data"):
        """
        Initialize the DataPipeline
    
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.loader = DataLoader(self.input_path)
        self.cleaner = DataCleaner()
        self.encoder = FeatureEncoder()
        self.splitter = DataSplitter()
        
    def run(self):
        """Run the complete data processing pipeline"""
        try:
            # Load data
            df = self.loader.load_data()
            
            # Clean data
            df_clean = self.cleaner.fit_transform(df)
            
            # Encode features
            df_encoded = self.encoder.fit_transform(df_clean)
            
            # Split features and target
            X, y = self.splitter.split_features_target(df_encoded)
            
            # Split into train/test
            X_train, X_test, y_train, y_test = self.splitter.train_test_split(X, y)
            
            # Save outputs
            self._save_outputs(
                df_clean=df_clean,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )
            
            logger.info("🏁 Data preprocessing complete.")
            return df_clean, X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _save_outputs(self, 
                     df_clean: pd.DataFrame,
                     X_train: np.ndarray,
                     X_test: np.ndarray,
                     y_train: pd.Series,
                     y_test: pd.Series) -> None:
        """Save all processed data and artifacts"""
        logger.info("💾 Saving cleaned data, scaler, and splits...")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned dataset
        df_clean.to_csv(self.output_dir / "cleaned_listings.csv", index=False)
        
        # Save train/test splits
        pd.DataFrame(X_train, columns=self.splitter.feature_columns_).to_csv(
            self.output_dir / "X_train_scaled.csv", index=False)
        pd.DataFrame(X_test, columns=self.splitter.feature_columns_).to_csv(
            self.output_dir / "X_test_scaled.csv", index=False)
        y_train.to_csv(self.output_dir / "y_train.csv", index=False)
        y_test.to_csv(self.output_dir / "y_test.csv", index=False)
        
        # Save scaler if it exists
        if self.splitter.scaler is not None:
            joblib.dump(self.splitter.scaler, self.output_dir / "scaler.pkl")
        
        # Save feature names
        with open(self.output_dir / "feature_names.txt", "w") as f:
            f.write("\n".join(self.splitter.feature_columns_))

if __name__ == "__main__":
    # Configuration
    RAW_DATA_PATH = os.path.join("data", "London_Listings.csv")
    OUTPUT_DIR = os.path.join("data", "processed")
    
    # Run pipeline
    pipeline = DataPipeline(input_path=RAW_DATA_PATH, output_dir=OUTPUT_DIR)
    pipeline.run()