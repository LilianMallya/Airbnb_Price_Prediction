import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
from pandas.plotting import scatter_matrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDAAnalyzer:
    def __init__(self, data_path: str = "data/processed/cleaned_listings.csv", 
                 output_dir: str = "outputs"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.df = None
        self._setup_visuals()

    def _setup_visuals(self):
        plt.style.use('seaborn-v0_8')  
        sns.set_palette("viridis")
        plt.rcParams['figure.dpi'] = 120
        plt.rcParams['savefig.dpi'] = 300

    def load_data(self):
        logger.info(f"Loading data from {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded. Shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def create_output_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self):
        try:
            self.create_output_dir()
            self.load_data()

            self.plot_price_distributions()
            self.plot_room_type_analysis()

            if {'latitude', 'longitude'}.issubset(self.df.columns):
                self.plot_geographic_distributions()

            self.plot_correlation_analysis()
            self.plot_scatter_matrix()
            self.plot_feature_relationships()
            self.plot_categorical_relationships()

            logger.info(f"✅ EDA complete. Results saved to {self.output_dir}")

        except Exception as e:
            logger.error(f"EDA failed: {e}")
            raise

    def plot_price_distributions(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.histplot(self.df['price'], bins=100, kde=True, ax=axes[0])
        axes[0].set_title('Original Price Distribution')
        axes[0].set_xlabel('Price (£)')
        axes[0].set_xlim(0, self.df['price'].quantile(0.99))

        if 'log_price' in self.df.columns:
            sns.histplot(self.df['log_price'], bins=100, kde=True, 
                         color='orange', ax=axes[1])
            axes[1].set_title('Log-Transformed Price Distribution')
            axes[1].set_xlabel('log(Price + 1)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'price_distributions.png')
        plt.close()

    def plot_room_type_analysis(self):
        if 'room_type' not in self.df.columns:
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.boxplot(data=self.df, x='room_type', y='price', ax=axes[0])
        axes[0].set_title('Price by Room Type')
        axes[0].set_ylim(0, self.df['price'].quantile(0.95))
        axes[0].tick_params(axis='x', rotation=45)

        sns.countplot(data=self.df, x='room_type', 
                      order=self.df['room_type'].value_counts().index,
                      ax=axes[1])
        axes[1].set_title('Room Type Distribution')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'room_type_analysis.png')
        plt.close()

    def plot_geographic_distributions(self):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.df, x='longitude', y='latitude',
                        hue='price', size='price', sizes=(20, 200),
                        alpha=0.6, palette='viridis')
        plt.title('Price Distribution by Location')
        plt.savefig(self.output_dir / 'geo_price_distribution.png')
        plt.close()

        if 'room_type' in self.df.columns:
            plt.figure(figsize=(12, 8))
            sns.scatterplot(data=self.df, x='longitude', y='latitude',
                            hue='room_type', alpha=0.6)
            plt.title('Room Type Distribution by Location')
            plt.savefig(self.output_dir / 'geo_room_types.png')
            plt.close()

    def plot_correlation_analysis(self):
        numeric_df = self.df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                        cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'correlation_matrix.png')
            plt.close()

            if 'log_price' in numeric_df.columns:
                corrs = numeric_df.corr()['log_price'].drop('log_price')
                plt.figure(figsize=(10, 6))
                sns.barplot(x=corrs.values, y=corrs.index, palette='rocket')
                plt.title('Feature Correlation with Log Price')
                plt.savefig(self.output_dir / 'feature_correlations.png')
                plt.close()

    def plot_scatter_matrix(self):
        numeric_df = self.df.select_dtypes(include=[np.number])
        selected_features = ['log_price', 'accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
        
        if all(f in numeric_df.columns for f in selected_features):
            subset = numeric_df[selected_features].copy()
            scatter_matrix(subset, alpha=0.2, figsize=(12, 10), diagonal='kde')
            plt.suptitle('Scatter Matrix of Key Numeric Features')
            plt.savefig(self.output_dir / 'scatter_matrix.png')
            plt.close()

    def plot_feature_relationships(self):
        if 'bedrooms' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.regplot(data=self.df, x='bedrooms', y='price',
                        scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            plt.ylim(0, self.df['price'].quantile(0.95))
            plt.title('Price vs Number of Bedrooms')
            plt.savefig(self.output_dir / 'bedrooms_vs_price.png')
            plt.close()

        if 'accommodates' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x='accommodates', y='price')
            plt.ylim(0, self.df['price'].quantile(0.95))
            plt.title('Price by Accommodation Capacity')
            plt.savefig(self.output_dir / 'accommodates_vs_price.png')
            plt.close()

    def plot_categorical_relationships(self):
        if 'neighbourhood' in self.df.columns:
            top_neigh = (self.df.groupby('neighbourhood')['price']
                         .median()
                         .sort_values(ascending=False)
                         .head(10))

            plt.figure(figsize=(12, 6))
            sns.barplot(x=top_neigh.values, y=top_neigh.index)
            plt.title('Top 10 Neighborhoods by Median Price')
            plt.savefig(self.output_dir / 'top_neighborhoods.png')
            plt.close()

            plt.figure(figsize=(12, 6))
            sns.boxplot(data=self.df[self.df['neighbourhood'].isin(top_neigh.index)],
                        x='price', y='neighbourhood', 
                        order=top_neigh.index)
            plt.xlim(0, self.df['price'].quantile(0.95))
            plt.title('Price Distribution in Top Neighborhoods')
            plt.savefig(self.output_dir / 'neighborhood_price_dist.png')
            plt.close()

if __name__ == "__main__":
    analyzer = EDAAnalyzer()
    analyzer.run_analysis()
