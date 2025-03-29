import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Tuple, Dict, List
import logging
from pathlib import Path
import json, joblib
from kneed import KneeLocator  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/kmeans_clustering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClusterDataLoader:
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.features = [
            'latitude', 'longitude', 'minimum_nights',
            'calculated_host_listings_count', 'availability_365'
        ]
        
    def load_and_preprocess(self) -> pd.DataFrame:
        logger.info("Loading and preprocessing data for clustering...")
        try:
            df = pd.read_csv(self.data_dir / "cleaned_listings.csv")
            
            df = pd.get_dummies(df, columns=['room_type'], drop_first=True)
            for col in ['room_type_Private room', 'room_type_Shared room']:
                if col not in df.columns:
                    df[col] = 0
            
            cluster_features = self.features + ['room_type_Private room', 'room_type_Shared room']
            existing_features = [col for col in cluster_features if col in df.columns]
            missing_features = [col for col in cluster_features if col not in df.columns]
            
            if missing_features:
                logger.warning(f"Missing features excluded: {missing_features}")
            
            return df[existing_features].fillna(0)
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

class ClusterFeatureEngineer:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        
    def preprocess(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(X)
    
    def reduce_dimensionality(self, X: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X)
    
    def save_artifacts(self, output_dir: Path) -> None:
        """Save preprocessing objects"""
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, output_dir / "cluster_scaler.pkl")
        joblib.dump(self.pca, output_dir / "cluster_pca.pkl")

class OptimalClusterFinder:
    """Determines optimal number of clusters """
    
    def __init__(self, max_k: int = 10):
        self.max_k = max_k
        self.metrics = {}
        
    def find_optimal_k(self, X: np.ndarray) -> int:
        """Determine optimal k """
        logger.info("Determining optimal number of clusters...")
        
        # Calculate metrics for different k values
        wcss = []
        silhouettes = []
        db_scores = []
        ch_scores = []
        
        for k in range(2, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            wcss.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            db_scores.append(davies_bouldin_score(X, labels))
            ch_scores.append(calinski_harabasz_score(X, labels))
        
        # Store metrics
        self.metrics = {
            'wcss': wcss,
            'silhouette': silhouettes,
            'davies_bouldin': db_scores,
            'calinski_harabasz': ch_scores
        }
        
        # Find best k 
        best_k_silhouette = np.argmax(silhouettes) + 2
        best_k_db = np.argmin(db_scores) + 2
        best_k_ch = np.argmax(ch_scores) + 2
        
        # Use silhouette
        optimal_k = best_k_silhouette
        
        logger.info(f"Optimal clusters - Silhouette: {best_k_silhouette}, "
                   f"Davies-Bouldin: {best_k_db}, Calinski-Harabasz: {best_k_ch}")
        
        return optimal_k
    
    def plot_metrics(self, output_path: Path) -> None:
        """Visualize cluster quality metrics"""
        plt.figure(figsize=(12, 8))
        
        # WCSS
        plt.subplot(2, 2, 1)
        plt.plot(range(2, self.max_k + 1), self.metrics['wcss'], 'bo-')
        plt.title('Within-Cluster Sum of Squares')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        
        # Silhouette Score
        plt.subplot(2, 2, 2)
        plt.plot(range(2, self.max_k + 1), self.metrics['silhouette'], 'go-')
        plt.title('Silhouette Score')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        
        # Davies-Bouldin
        plt.subplot(2, 2, 3)
        plt.plot(range(2, self.max_k + 1), self.metrics['davies_bouldin'], 'ro-')
        plt.title('Davies-Bouldin Index')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        
        # Calinski-Harabasz
        plt.subplot(2, 2, 4)
        plt.plot(range(2, self.max_k + 1), self.metrics['calinski_harabasz'], 'mo-')
        plt.title('Calinski-Harabasz Score')
        plt.xlabel('Number of clusters')
        plt.ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class ClusterVisualizer:
    
    @staticmethod
    def visualize_clusters(X_reduced: np.ndarray, labels: np.ndarray, 
                         output_path: Path) -> None:
        """Create 2D visualization of clusters"""
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=X_reduced[:, 0], 
            y=X_reduced[:, 1], 
            hue=labels, 
            palette='viridis', 
            s=60, 
            edgecolor='k',
            alpha=0.7
        )
        plt.title("PCA Projection of K-Means Clusters")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class KMeansPipeline:
    """complete KMeans clustering pipeline"""
    
    def __init__(self):
        self.data_loader = ClusterDataLoader()
        self.feature_engineer = ClusterFeatureEngineer()
        self.cluster_finder = OptimalClusterFinder(max_k=10)
        self.visualizer = ClusterVisualizer()
        self.model = None
        
    def run(self):
        """Execute the complete clustering pipeline"""
        try:
            X = self.data_loader.load_and_preprocess()
            X_scaled = self.feature_engineer.preprocess(X)
            
            optimal_k = self.cluster_finder.find_optimal_k(X_scaled)
            
            # Train final model
            self.model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            labels = self.model.fit_predict(X_scaled)
            
            # Reduce dimensions for visualization
            X_reduced = self.feature_engineer.reduce_dimensionality(X_scaled)
            
            self._save_results(X, labels)
            
            # Visualizations
            self.cluster_finder.plot_metrics(Path("outputs/cluster_metrics.png"))
            self.visualizer.visualize_clusters(
                X_reduced, labels, 
                Path("outputs/kmeans_clusters.png")
            )
            
            logger.info(f"""
            ✅ Clustering Complete
            =====================
            Optimal clusters: {optimal_k}
            Cluster sizes: {np.bincount(labels)}
            """)
            
        except Exception as e:
            logger.error(f"Clustering pipeline failed: {str(e)}")
            raise
    
    def _save_results(self, original_data: pd.DataFrame, labels: np.ndarray) -> None:
        """Save all clustering results"""
        # Create directories if needed
        Path("outputs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        
        clustered_data = original_data.copy()
        clustered_data['cluster'] = labels
        clustered_data.to_csv(Path("data/clustered_listings.csv"), index=False)
        
        joblib.dump(self.model, Path("models/kmeans_model.pkl"))
        self.feature_engineer.save_artifacts(Path("models"))
        
        with open(Path("outputs/cluster_metrics.json"), "w") as f:
            json.dump({
                'optimal_k': int(self.model.n_clusters),
                'inertia': float(self.model.inertia_),
                'silhouette_score': float(silhouette_score(
                    self.feature_engineer.scaler.transform(original_data), 
                    labels
                ))
            }, f, indent=4)

if __name__ == "__main__":
    pipeline = KMeansPipeline()
    pipeline.run()