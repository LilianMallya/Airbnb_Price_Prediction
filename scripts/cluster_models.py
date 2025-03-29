import os
import json
import pandas as pd
from typing import List, Dict
from pathlib import Path
import logging
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/cluster_models.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ClusterModelConfig:
    """Configuration"""
    cluster_col: str = 'cluster'
    target_col: str = 'minimum_nights'
    min_cluster_size: int = 20
    categorical_cols: List[str] = field(default_factory=lambda: [
        'room_type_Private room',
        'room_type_Shared room'
    ])
    numeric_cols: List[str] = field(default_factory=lambda: [
        'calculated_host_listings_count'
    ])
    model_types: List[str] = field(default_factory=lambda: ['linear', 'rf'])
    cv_folds: int = 3
    seed: int = 42

class SparkClusterModeler:
    """Cluster-based modeling """
    
    def __init__(self, config: ClusterModelConfig = ClusterModelConfig()):
        self.config = config
        self.spark = self._init_spark()
        self.models = {}
        self.metrics = {}
        
    def _init_spark(self):
        """Initialize Spark """
        return SparkSession.builder \
            .appName("AirbnbClusterModeling") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.sql.shuffle.partitions", "50") \
            .getOrCreate()
    
    def load_data(self, data_path: str):
        """Load and validate the dataset"""
        logger.info(f"Loading data from {data_path}")
        try:
            pdf = pd.read_csv(data_path)
            
            # Verify required columns exist
            required_cols = {self.config.cluster_col, self.config.target_col}
            missing_cols = required_cols - set(pdf.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            df = self.spark.createDataFrame(pdf)
            logger.info(f"Data loaded with shape: {pdf.shape}")
            logger.info("Cluster distribution:")
            df.groupBy(self.config.cluster_col).count().show()
            
            return df
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _preprocess_for_cluster(self, df, cluster_id: int):
        """Prepare data for a specific cluster"""
        cluster_df = df.filter(col(self.config.cluster_col) == cluster_id)
        
        if cluster_df.count() < self.config.min_cluster_size:
            logger.warning(f"Skipping cluster {cluster_id} - only {cluster_df.count()} samples")
            return None, None, None
        
        # Feature engineering pipeline
        stages = []
        all_features = self.config.numeric_cols + self.config.categorical_cols
        
        assembler = VectorAssembler(inputCols=all_features, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        stages += [assembler, scaler]
        
        pipeline = Pipeline(stages=stages)
        preprocessor = pipeline.fit(cluster_df)
        processed_df = preprocessor.transform(cluster_df)
        
        return processed_df, preprocessor, all_features
    
    def _train_model(self, df, features: List[str], cluster_id: int) -> Dict:
        """Train and evaluate models for a cluster"""
        cluster_metrics = {}
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=self.config.seed)
        
        models = {
            'linear': LinearRegression(
                featuresCol="scaled_features", 
                labelCol=self.config.target_col,
                elasticNetParam=0.5
            ),
            'rf': RandomForestRegressor(
                featuresCol="scaled_features",
                labelCol=self.config.target_col,
                numTrees=50,
                maxDepth=5,
                seed=self.config.seed
            )
        }
        
        for model_type in self.config.model_types:
            if model_type not in models:
                continue
                
            logger.info(f"Training {model_type.upper()} for cluster {cluster_id}")
            
            param_grid = self._get_param_grid(model_type)
            evaluator = RegressionEvaluator(
                labelCol=self.config.target_col,
                predictionCol="prediction",
                metricName="rmse"
            )
            
            cv = CrossValidator(
                estimator=models[model_type],
                estimatorParamMaps=param_grid,
                evaluator=evaluator,
                numFolds=self.config.cv_folds,
                seed=self.config.seed
            )
            
            cv_model = cv.fit(train_df)
            best_model = cv_model.bestModel
            
            test_preds = best_model.transform(test_df)
            rmse = evaluator.evaluate(test_preds)
            r2 = evaluator.setMetricName("r2").evaluate(test_preds)
            
            cluster_metrics[model_type] = {
                'rmse': rmse,
                'r2': r2,
                'params': self._serialize_params(best_model.extractParamMap()),
                'feature_importance': self._get_feature_importance(best_model, features, model_type)
            }
            
            self._save_model_artifacts(best_model, cluster_id, model_type)
            
        return cluster_metrics
    
    def _serialize_params(self, param_map: Dict) -> Dict:
        """Convert PySpark Param objects"""
        return {str(k.name): str(v) for k, v in param_map.items()}
    
    def _get_param_grid(self, model_type: str):
        """hyperparameter grid for each model type"""
        if model_type == 'linear':
            return ParamGridBuilder() \
                .addGrid(LinearRegression.elasticNetParam, [0.0, 0.5, 1.0]) \
                .addGrid(LinearRegression.regParam, [0.01, 0.1]) \
                .build()
        elif model_type == 'rf':
            return ParamGridBuilder() \
                .addGrid(RandomForestRegressor.maxDepth, [3, 5]) \
                .addGrid(RandomForestRegressor.numTrees, [30, 50]) \
                .build()
        return []
    
    def _get_feature_importance(self, model, features: List[str], model_type: str) -> Dict:
        """Extract feature importance based on model type"""
        if model_type == 'linear':
            coeffs = model.coefficients.toArray().tolist()
            return dict(zip(features, coeffs))
        elif model_type == 'rf':
            importances = model.featureImportances.toArray().tolist()
            return dict(zip(features, importances))
        return {}
    
    def _save_model_artifacts(self, model, cluster_id: int, model_type: str) -> None:
        """Save model artifacts to disk"""
        model_dir = Path(f"models/cluster_{cluster_id}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model.write().overwrite().save(str(model_dir / f"spark_{model_type}_model"))
        
        with open(model_dir / f"{model_type}_metrics.json", "w") as f:
            json.dump({
                'cluster_id': cluster_id,
                'model_type': model_type,
                'params': self._serialize_params(model.extractParamMap())
            }, f, indent=4)
    
    def train_all_clusters(self, data_path: str) -> None:
        """Train models for all clusters"""
        try:
            os.makedirs("outputs", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            
            df = self.load_data(data_path)
            cluster_ids = [row[self.config.cluster_col] 
                         for row in df.select(self.config.cluster_col).distinct().collect()]
            
            for cluster_id in cluster_ids:
                logger.info(f"\n{'='*50}\nProcessing cluster {cluster_id}\n{'='*50}")
                
                processed_df, preprocessor, features = self._preprocess_for_cluster(df, cluster_id)
                if processed_df is None:
                    continue
                
                cluster_metrics = self._train_model(processed_df, features, cluster_id)
                self.metrics[cluster_id] = cluster_metrics
                
                preprocessor.write().overwrite().save(
                    str(Path(f"models/cluster_{cluster_id}/preprocessor")))
            
            self._save_global_metrics()
            self._generate_comparison_plots()
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            self.spark.stop()
    
    def _save_global_metrics(self) -> None:
        """Save metrics """
        records = []
        for cluster_id, models in self.metrics.items():
            for model_type, metrics in models.items():
                records.append({
                    'cluster': str(cluster_id),  
                    'model_type': model_type,
                    'rmse': float(metrics['rmse']),
                    'r2': float(metrics['r2']),
                    'n_features': int(len(metrics['feature_importance']))
                })
        
        pd.DataFrame(records).to_csv("outputs/cluster_model_metrics.csv", index=False)
        
        # Create metrics 
        serializable_metrics = {}
        for cluster_id, models in self.metrics.items():
            serializable_metrics[str(cluster_id)] = {
                model_type: {
                    'rmse': float(metrics['rmse']),
                    'r2': float(metrics['r2']),
                    'params': metrics['params'],  
                    'feature_importance': {
                        k: float(v) for k, v in metrics['feature_importance'].items()
                    }
                }
                for model_type, metrics in models.items()
            }
        
        with open("outputs/feature_importances.json", "w") as f:
            json.dump(serializable_metrics, f, indent=4)
    
    def _generate_comparison_plots(self) -> None:
        """Generate comparison plots"""
        try:
            metrics_df = pd.read_csv("outputs/cluster_model_metrics.csv")
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=metrics_df, x='model_type', y='r2')
            plt.title("Model Performance Comparison (R²)")
            plt.savefig("outputs/model_comparison.png")
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=metrics_df, x='model_type', y='rmse')
            plt.title("Model Performance Comparison (RMSE)")
            plt.savefig("outputs/model_comparison_rmse.png")
            plt.close()
            
            pivot_df = metrics_df.pivot(index='cluster', columns='model_type', values='r2')
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", center=0)
            plt.title("Cluster Performance Heatmap (R²)")
            plt.tight_layout()
            plt.savefig("outputs/cluster_heatmap.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")

if __name__ == "__main__":
    config = ClusterModelConfig(
        target_col='minimum_nights',
        categorical_cols=['room_type_Private room', 'room_type_Shared room'],
        numeric_cols=['calculated_host_listings_count'],
        model_types=['linear', 'rf'],
        min_cluster_size=20
    )
    
    modeler = SparkClusterModeler(config)
    modeler.train_all_clusters("data/clustered_listings.csv")
    
    print("\n✅ Modeling complete. Results saved in:")
    print("- outputs/cluster_model_metrics.csv")
    print("- outputs/feature_importances.json")
    print("- outputs/*.png (visualizations)")
    print("- models/ (trained models)")