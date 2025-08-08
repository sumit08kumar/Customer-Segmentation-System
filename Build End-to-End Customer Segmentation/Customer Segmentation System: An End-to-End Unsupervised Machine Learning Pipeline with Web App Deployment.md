# Customer Segmentation System: An End-to-End Unsupervised Machine Learning Pipeline with Web App Deployment

This project implements an end-to-end unsupervised machine learning system to segment customers based on behavioral and demographic data. It showcases multiple clustering algorithms, dimensionality reduction techniques, and an interactive Streamlit app for dynamic cluster analysis — all containerized and ready for cloud deployment.

## Project Summary:

This project aims to group customers into similar segments to help businesses personalize marketing, design targeted promotions, and improve customer retention strategies.

## Dataset:

*   **Source**: [Kaggle – Mall Customer Segmentation Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)
*   **Features**:
    *   `CustomerID`
    *   `Gender`
    *   `Age`
    *   `Annual Income`
    *   `Spending Score`

## Key Steps:

1.  **Exploratory Data Analysis (EDA)**: Histograms, pairplots, violin plots, and correlation matrix.
2.  **Preprocessing**: One-hot encoding for `Gender`, standardization of numerical features, and dimensionality reduction with PCA or t-SNE.
3.  **Unsupervised Model Comparison**: K-Means, Hierarchical Clustering (Agglomerative), DBSCAN, and Gaussian Mixture Models (GMM).
4.  **Evaluation Metrics**: Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Score, Elbow method & Dendrogram.
5.  **Visualization**: 2D PCA/t-SNE cluster plots and interactive Streamlit UI.

## Deployment:

*   **Streamlit Web App**: Input sliders, display cluster characteristics, visualize cluster plots and statistics.
*   **Dockerized Deployment**: Dockerfile to containerize the app for deployment.



