# Pricing Airbnb Listings as Heterogeneous Urban Markets

This project examines whether Airbnb listing prices in London can be modeled more accurately by treating the market as a set of distinct sub-markets rather than as a single global population.

Instead of focusing purely on prediction performance, the work explores how segmentation changes both model behaviour and interpretation when dealing with heterogeneous urban data.

---

## Summary

A single global pricing model performs poorly because it forces one relationship between features and price across listings that behave very differently in practice.

When listings are first grouped into clusters with similar characteristics, both predictive performance and residual behaviour improve. Feature importance also varies meaningfully across clusters, indicating that different types of listings are driven by different pricing dynamics.

---

## What Was Done

- Cleaned and preprocessed a real-world Airbnb dataset containing missing values, skewed prices, and mixed numerical and categorical features.
- Established baseline global models using linear regression and tree-based methods.
- Applied K-Means clustering to segment listings into groups with similar pricing behaviour.
- Trained separate models per cluster using PySpark.
- Compared global and cluster-based models using RMSE, R², residual plots, and feature importance analysis.

The emphasis was on testing whether segmentation addressed a structural issue in the data rather than simply improving fit.

---

## Observations

- Global models average over structurally different listings, leading to weaker performance and unstable residuals.
- Cluster-based models reduce error by training on more homogeneous subsets of data.
- Feature importance is not consistent across the market. Location dominates pricing in some clusters, while capacity and room configuration are more influential in others.
- Improvements persist across multiple model types, suggesting that the effect comes from segmentation rather than model choice.

---

## Trade-offs and Limitations

- Model performance is sensitive to the number of clusters chosen.
- Smaller clusters can overfit despite improved aggregate metrics.
- Geographic features risk leakage if not handled carefully.
- Segmentation increases system complexity and may not be appropriate for low-data settings.

These constraints mean segmentation should be applied selectively rather than by default.

---

## Repository Structure

- `scripts/` — data preprocessing, modeling, and evaluation pipelines  
- `models/` — trained models and preprocessing artifacts  
- `outputs/` — metrics and visualizations used for evaluation  
- `docs/` — setup instructions and execution details  

Instructions for running the project end-to-end are provided in **`docs/README.md`**.
