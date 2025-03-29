## Script: `data_preprocessing.py`

### Input:
- **File**: `data/London_Listings.csv`  
- **Format**: Raw Airbnb dataset with mixed types, missing values, and outliers.

### Output:
All output files are saved to the folder: `data/processed/`

- `cleaned_listings.csv`: Cleaned and preprocessed dataset  
- `X_train_scaled.csv`, `X_test_scaled.csv`: Scaled numerical features for training/testing  
- `y_train.csv`, `y_test.csv`: Corresponding target values  
- `scaler.pkl`: Saved `StandardScaler` object used for transformation  
- `feature_names.txt`: List of numerical features used in modeling

### Notes:
Run this script **before any modeling or analysis**. It prepares all necessary inputs for training and evaluation.


## `eda_visuals.py` – Exploratory Data Analysis Script

**Input**  
- `data/processed/cleaned_listings.csv`: Cleaned Airbnb listings dataset used for generating EDA plots.

**Output**  
Plots are saved automatically to the `outputs/` directory:
- `price_distributions.png`
- `room_type_analysis.png`
- `geo_price_distribution.png`
- `geo_room_types.png`
- `correlation_matrix.png`
- `feature_correlations.png`
- `scatter_matrix.png`
- `bedrooms_vs_price.png`
- `accommodates_vs_price.png`
- `top_neighborhoods.png`
- `neighborhood_price_dist.png`


## `baseline_linear_model.py` – Linear Regression Training Pipeline

**Input**  
- `data/processed/cleaned_listings.csv`

**Output**  
- Trained linear regression model saved to `models/linear_model.pkl`
- Model evaluation metrics saved to `models/model_metrics.json`
- Residual analysis plots saved to `outputs/residual_analysis.png`

## `random_forest_model.py` – Random Forest Regression Pipeline

**Input**  
- `data/processed/cleaned_listings.csv` 

**Output**  
- Trained model saved to: `models/random_forest_model.pkl`
- Scaler used in preprocessing saved to: `models/rf_scaler.pkl`
- Evaluation metrics saved to: `outputs/rf_metrics.json`
- Feature importance plot saved to: `outputs/rf_feature_importance.png`


## `kmeans_clustering.py` – K-Means Clustering Pipeline

**Input**  
- `data/processed/cleaned_listings.csv`

**Output**  
- Clustered dataset saved to: `data/clustered_listings.csv`
- Trained model saved to: `models/kmeans_model.pkl`
- Preprocessing artifacts saved to: `models/cluster_scaler.pkl`, `models/cluster_pca.pkl`
- Cluster evaluation metrics saved to: `outputs/cluster_metrics.json`
- Evaluation plots saved to:
  - `outputs/cluster_metrics.png` (Silhouette, Davies-Bouldin, Calinski-Harabasz, WCSS)
  - `outputs/kmeans_clusters.png` (PCA projection of clusters)

  ## `cluster_models.py` – Cluster-Based Modeling with PySpark

**Input**  
- `data/clustered_listings.csv`

**Output**  
- Trained models saved under: `models/cluster_{cluster_id}/spark_{model_type}_model`
- Preprocessing pipelines saved under: `models/cluster_{cluster_id}/preprocessor`
- Cluster-specific metrics:
  - `outputs/cluster_model_metrics.csv`: R², RMSE, and feature count per model and cluster
  - `outputs/feature_importances.json`: Feature importance breakdown per cluster and model
- Visualizations:
  - `outputs/model_comparison.png`: R² by model type
  - `outputs/model_comparison_rmse.png`: RMSE by model type
  - `outputs/cluster_heatmap.png`: Cluster-level R² scores in heatmap


  ## `evaluation_metrics.py` – Global vs Cluster-Based Model Evaluation (PySpark)

**Input**
- `data/processed/cleaned_listings.csv`
- `models/feature_names.pkl`: List of selected features (auto-generated if missing).
- Trained models (if previously saved) are loaded from `models/`.

**Output**
- Evaluation metrics for global models (Random Forest, Gradient Boosting, Linear Regression) and cluster-specific models.
- Visualizations saved under `reports/figures/`:
  - `cluster_distribution.png`: Bar chart of number of listings per cluster.
  - `model_comparison.png`: Side-by-side comparison of RMSE and R² for global vs cluster models.