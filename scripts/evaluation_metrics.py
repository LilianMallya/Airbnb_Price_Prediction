import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

class BaseEvaluator(ABC):

    
    def __init__(self, spark_session: SparkSession):
        
        self.spark = spark_session
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Configure paths
        self.data_dir = os.path.join(self.base_dir, "data", "processed")
        self.models_dir = os.path.join(self.base_dir, "models")
        self.reports_dir = os.path.join(self.base_dir, "reports", "figures")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
    def get_feature_names(self, df: DataFrame) -> List[str]:
        """Get or create feature names from dataframe with validation"""
        feature_path = os.path.join(self.models_dir, "feature_names.pkl")
        
        if os.path.exists(feature_path):
            try:
                features = joblib.load(feature_path)
                # Validate that all features exist in the dataframe
                missing = [f for f in features if f not in df.columns]
                if not missing:
                    return features
                print(f"⚠️ Some features in feature_names.pkl not found in data: {missing}")
            except Exception as e:
                print(f"⚠️ Error loading feature names: {str(e)}")
        
        print("⚠️ feature_names.pkl not found or invalid - inferring features from dataframe")
        
       
        NON_FEATURES = {
            'price', 'log_price', 'cluster', 'id', 'listing_id',
            'host_id', 'neighbourhood', 'room_type', 'property_type', 'prediction'
        }
        
        # Get numeric columns 
        numeric_cols = [f.name for f in df.schema.fields 
                       if f.dataType.typeName() in ['integer', 'double', 'float']]
        
        features = [col for col in numeric_cols if col not in NON_FEATURES]
        
        # Save the inferred features
        joblib.dump(features, feature_path)
        print(f"✅ Created feature_names.pkl with {len(features)} features")
        return features

    def preprocess_data(self, df: DataFrame, feature_names: List[str]) -> DataFrame:
        """Handle null values """
        try:
            
            imputers = [Imputer(inputCol=col, outputCol=f"{col}_imputed") 
                      for col in feature_names]
            
            
            imputation_pipeline = Pipeline(stages=imputers)
            df = imputation_pipeline.fit(df).transform(df)
            
           
            assembler = VectorAssembler(
                inputCols=[f"{col}_imputed" for col in feature_names],
                outputCol="features",
                handleInvalid="keep"
            )
            
            return assembler.transform(df)
        except Exception as e:
            print(f"❌ Preprocessing failed: {str(e)}")
            raise

    @abstractmethod
    def load_data(self) -> DataFrame:
        """Load and prepare data for evaluation"""
        pass
    
    @abstractmethod
    def evaluate(self, data: DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance"""
        pass


class ClusterBasedEvaluator(BaseEvaluator):
    """Evaluator that creates and evaluates cluster-specific models"""
    
    def __init__(self, spark_session: SparkSession):
        super().__init__(spark_session)
        self.num_clusters = 3
        self.cluster_col = "cluster_assignment"  

    def create_kmeans_model(self, data: DataFrame) -> DataFrame:
        """Create and save KMeans model if it doesn't exist"""
        kmeans_path = os.path.join(self.models_dir, "kmeans_model")
        
        if os.path.exists(kmeans_path):
            try:
                model = KMeansModel.load(kmeans_path)
                return model.transform(data).withColumnRenamed("prediction", self.cluster_col)
            except Exception as e:
                print(f"⚠️ Error loading KMeans model: {str(e)} - training new model")
                
        print("⚠️ KMeans model not found - training new model")
        kmeans = KMeans(featuresCol="features", k=self.num_clusters)
        model = kmeans.fit(data)
        model.save(kmeans_path)
        print(f"✅ Created KMeans model with {self.num_clusters} clusters")
        return model.transform(data).withColumnRenamed("prediction", self.cluster_col)
    
    def create_cluster_model(self, cluster_id: int, cluster_data: DataFrame) -> PipelineModel:
        model_path = os.path.join(self.models_dir, f"cluster_{cluster_id}_model")
        
        if os.path.exists(model_path):
            try:
                return PipelineModel.load(model_path)
            except Exception as e:
                print(f"⚠️ Error loading cluster {cluster_id} model: {str(e)} - training new model")
                
        print(f"⚠️ Cluster {cluster_id} model not found - training new model")
        rf = RandomForestRegressor(
            featuresCol="features", 
            labelCol="log_price",
            predictionCol="price_prediction"  
        )
        pipeline = Pipeline(stages=[rf])
        model = pipeline.fit(cluster_data)
        model.save(model_path)
        print(f"✅ Created cluster {cluster_id} model")
        return model

    def load_data(self) -> DataFrame:
        print("📦 Loading and preparing clustered data...")
        
        try:
            
            data_path = os.path.join(self.data_dir, "cleaned_listings.csv")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Cleaned listings data not found at {data_path}")
                
            df = self.spark.read.csv(data_path, header=True, inferSchema=True)
           
            required_cols = {'log_price'}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}") 
            
            feature_names = self.get_feature_names(df)
           
            df = self.preprocess_data(df, feature_names)
            
            df = self.create_kmeans_model(df)
            return df
            
        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            raise
            
    def evaluate(self, data: Optional[DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all cluster models """
        if data is None:
            data = self.load_data()
            
        results = {}
        
        for cluster_id in range(self.num_clusters):
            try:
                cluster_data = data.filter(data[self.cluster_col] == cluster_id)
                sample_count = cluster_data.count()
                
                if sample_count == 0:
                    print(f"⚠️ No data found for cluster {cluster_id}")
                    results[f"cluster_{cluster_id}"] = {
                        "rmse": np.nan,
                        "r2": np.nan,
                        "mae": np.nan,
                        "samples": 0
                    }
                    continue
                    
                model = self.create_cluster_model(cluster_id, cluster_data)
                predictions = model.transform(cluster_data)
                
                evaluator = RegressionEvaluator(
                    labelCol="log_price",
                    predictionCol="price_prediction",  
                    metricName="rmse"
                )
                
                metrics = {
                    "rmse": evaluator.evaluate(predictions),
                    "r2": evaluator.evaluate(predictions, {evaluator.metricName: "r2"}),
                    "mae": evaluator.evaluate(predictions, {evaluator.metricName: "mae"}),
                    "samples": sample_count
                }
                
                results[f"cluster_{cluster_id}"] = metrics
                print(f"✅ Cluster {cluster_id} evaluation completed - RMSE: {metrics['rmse']:.3f}")
                
            except Exception as e:
                print(f"⚠️ Evaluation failed for cluster {cluster_id}: {str(e)}")
                results[f"cluster_{cluster_id}"] = {
                    "rmse": np.nan,
                    "r2": np.nan,
                    "mae": np.nan,
                    "samples": 0
                }
                
        return results


class GlobalModelEvaluator(BaseEvaluator):
    """ creates and evaluates global models"""
    
    def __init__(self, spark_session: SparkSession):
        super().__init__(spark_session)
        self.models = {
            "Random Forest": RandomForestRegressor,
            "Gradient Boosting": GBTRegressor,
            "Linear Regression": LinearRegression
        }
        
    def create_model(self, name: str, model_class: Any, data: DataFrame) -> PipelineModel:
        """Create and save a global model if it doesn't exist"""
        model_path = os.path.join(self.models_dir, f"{name.lower().replace(' ', '_')}_model")
        
        if os.path.exists(model_path):
            try:
                return PipelineModel.load(model_path)
            except Exception as e:
                print(f"⚠️ Error loading {name} model: {str(e)} - training new model")
                
        print(f"⚠️ {name} model not found - training new model")
        model = model_class(
            featuresCol="features", 
            labelCol="log_price",
            predictionCol="price_prediction"  
        )
        pipeline = Pipeline(stages=[model])
        trained_model = pipeline.fit(data)
        trained_model.save(model_path)
        print(f"✅ Created {name} model")
        return trained_model

    def load_data(self) -> DataFrame:
        print("📦 Loading data for global models...")
        
        try:
            data_path = os.path.join(self.data_dir, "cleaned_listings.csv")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Cleaned listings data not found at {data_path}")
                
            df = self.spark.read.csv(data_path, header=True, inferSchema=True)
            
            # Validate required columns
            required_cols = {'log_price'}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            feature_names = self.get_feature_names(df)
            return self.preprocess_data(df, feature_names)
            
        except Exception as e:
            print(f"❌ Data loading failed: {str(e)}")
            raise
            
    def evaluate(self, data: Optional[DataFrame] = None) -> Dict[str, Dict[str, float]]:
        """Evaluate all global models"""
        if data is None:
            data = self.load_data()
            
        results = {}
        
        for name, model_class in self.models.items():
            try:
                model = self.create_model(name, model_class, data)
                predictions = model.transform(data)
                
                evaluator = RegressionEvaluator(
                    labelCol="log_price",
                    predictionCol="price_prediction",  
                    metricName="rmse"
                )
                
                metrics = {
                    "rmse": evaluator.evaluate(predictions),
                    "r2": evaluator.evaluate(predictions, {evaluator.metricName: "r2"}),
                    "mae": evaluator.evaluate(predictions, {evaluator.metricName: "mae"}),
                    "samples": data.count()
                }
                
                results[name] = metrics
                print(f"✅ {name} evaluation completed - RMSE: {metrics['rmse']:.3f}")
                
            except Exception as e:
                print(f"⚠️ Evaluation failed for {name}: {str(e)}")
                results[name] = {
                    "rmse": np.nan,
                    "r2": np.nan,
                    "mae": np.nan,
                    "samples": 0
                }
                
        return results


class ModelComparisonVisualizer:
    """ visualizer """
    
    @staticmethod
    def plot_cluster_distribution(data: DataFrame, save_path: str) -> None:
        """Plot cluster distribution """
        try:
            cluster_counts = data.groupBy("cluster_assignment").count().toPandas()
            
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                x="cluster_assignment", 
                y="count", 
                data=cluster_counts, 
                palette="viridis"
            )
            
            plt.title("Distribution of Listings Across Clusters", fontsize=14, pad=20)
            plt.xlabel("Cluster ID", fontsize=12)
            plt.ylabel("Number of Listings", fontsize=12)
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(
                    f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), 
                    textcoords='offset points'
                )
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved cluster distribution plot to {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to plot cluster distribution: {str(e)}")

    @staticmethod
    def plot_model_comparison(global_results: Dict, cluster_results: Dict, save_path: str) -> None:
        """ model comparison visualization """
        try:
            global_df = pd.DataFrame.from_dict(global_results, orient='index')
            cluster_df = pd.DataFrame.from_dict(cluster_results, orient='index')
            
            global_df = global_df.dropna()
            cluster_df = cluster_df.dropna()
            
            if global_df.empty or cluster_df.empty:
                raise ValueError("No valid results to plot")
            
            plt.figure(figsize=(14, 6))
            
            # RMSE comparison
            plt.subplot(1, 2, 1)
            sns.barplot(
                x=global_df.index, 
                y=global_df['rmse'], 
                color='#3498db', 
                label='Global', 
                alpha=0.7
            )
            sns.barplot(
                x=cluster_df.index, 
                y=cluster_df['rmse'], 
                color='#e74c3c', 
                label='Cluster', 
                alpha=0.7
            )
            plt.title("RMSE Comparison", fontsize=13, pad=15)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("RMSE (log price)")
            plt.legend()
            
            # R² comparison
            plt.subplot(1, 2, 2)
            sns.barplot(
                x=global_df.index, 
                y=global_df['r2'], 
                color='#3498db', 
                label='Global', 
                alpha=0.7
            )
            sns.barplot(
                x=cluster_df.index, 
                y=cluster_df['r2'], 
                color='#e74c3c', 
                label='Cluster', 
                alpha=0.7
            )
            plt.title("R² Comparison", fontsize=13, pad=15)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel("R² Score")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved model comparison plot to {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to plot model comparison: {str(e)}")


class AdvancedEvaluationSystem:
    """Main evaluation system """
    
    def __init__(self):
        """Initialize Spark """
        self.spark = SparkSession.builder \
            .appName("AirbnbPriceEvaluation") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "8") \
            .getOrCreate()
            
        self.cluster_evaluator = ClusterBasedEvaluator(self.spark)
        self.global_evaluator = GlobalModelEvaluator(self.spark)
        self.visualizer = ModelComparisonVisualizer()
        
    def run_evaluation(self) -> bool:
        """Run evaluation pipeline"""
        try:
            print("🚀 Starting advanced evaluation pipeline...\n")
            
            print("🔍 Loading and validating data...")
            data = self.cluster_evaluator.load_data()
            
            cluster_plot_path = os.path.join(
                self.cluster_evaluator.reports_dir, 
                "cluster_distribution.png"
            )
            print("\n📊 Visualizing cluster distribution...")
            self.visualizer.plot_cluster_distribution(data, cluster_plot_path)
            
            # Evaluate cluster models
            print("\n🔍 Evaluating cluster-based models...")
            cluster_results = self.cluster_evaluator.evaluate(data)
            print("\n📋 Cluster Model Results:")
            for cluster, metrics in cluster_results.items():
                print(f"  {cluster}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}, "
                      f"MAE={metrics['mae']:.3f}, Samples={metrics['samples']}")
            
            # Evaluate global models
            print("\n🔍 Evaluating global models...")
            global_results = self.global_evaluator.evaluate(data)
            print("\n📋 Global Model Results:")
            for model, metrics in global_results.items():
                print(f"  {model}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}, "
                      f"MAE={metrics['mae']:.3f}, Samples={metrics['samples']}")
            
            # Compare and visualize results
            comparison_plot_path = os.path.join(
                self.cluster_evaluator.reports_dir, 
                "model_comparison.png"
            )
            print("\n📊 Generating comparison visualizations...")
            self.visualizer.plot_model_comparison(global_results, cluster_results, comparison_plot_path)
            
            print("\n🎉 Evaluation completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {str(e)}")
            self.print_troubleshooting_guide()
            return False
            
        finally:
            self.spark.stop()
    


if __name__ == "__main__":
    print("\n" + "="*50)
    print("AIRBNB PRICE PREDICTION EVALUATION SYSTEM")
    print("="*50 + "\n")
    
    evaluation_system = AdvancedEvaluationSystem()
    success = evaluation_system.run_evaluation()
    
    if not success:
        exit(1)