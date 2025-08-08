import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/Mall_Customers.csv')
        df.rename(columns={'Annual Income (k$)': 'Annual Income', 'Spending Score (1-100)': 'Spending Score'}, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'mall_customers.csv' is in the 'data' directory.")
        return pd.DataFrame()

# Load preprocessor and models
@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        with open('models/preprocessor.pkl', 'rb') as f:
            artifacts['preprocessor'] = pickle.load(f)
        with open('models/kmeans.pkl', 'rb') as f:
            artifacts['kmeans'] = pickle.load(f)
        with open('models/agglomerativeclustering.pkl', 'rb') as f:
            artifacts['agglomerativeclustering'] = pickle.load(f)
        with open('models/dbscan.pkl', 'rb') as f:
            artifacts['dbscan'] = pickle.load(f)
        with open('models/gaussianmixture.pkl', 'rb') as f:
            artifacts['gaussianmixture'] = pickle.load(f)
        with open('models/pca.pkl', 'rb') as f:
            artifacts['pca'] = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"Model artifact not found: {e}. Please run the clustering notebook first.")
    return artifacts

df = load_data()
artifacts = load_artifacts()

if not df.empty and artifacts:
    preprocessor = artifacts.get('preprocessor')
    kmeans_model = artifacts.get('kmeans')
    agglomerative_model = artifacts.get('agglomerativeclustering')
    dbscan_model = artifacts.get('dbscan')
    gmm_model = artifacts.get('gaussianmixture')
    pca_model = artifacts.get('pca')

    st.title("🛍️ Customer Segmentation System")

    st.sidebar.header("Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select Clustering Model",
        ["KMeans", "AgglomerativeClustering", "DBSCAN", "GaussianMixture"]
    )

    st.sidebar.subheader("Filter Customers")
    min_age, max_age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    min_income, max_income = st.sidebar.slider("Annual Income (k$)", int(df['Annual Income'].min()), int(df['Annual Income'].max()), (int(df['Annual Income'].min()), int(df['Annual Income'].max())))
    min_spending, max_spending = st.sidebar.slider("Spending Score (1-100)", int(df['Spending Score'].min()), int(df['Spending Score'].max()), (int(df['Spending Score'].min()), int(df['Spending Score'].max())))
    selected_gender = st.sidebar.multiselect("Gender", df['Gender'].unique(), df['Gender'].unique())

    filtered_df = df[
        (df['Age'] >= min_age) & (df['Age'] <= max_age) &
        (df['Annual Income'] >= min_income) & (df['Annual Income'] <= max_income) &
        (df['Spending Score'] >= min_spending) & (df['Spending Score'] <= max_spending) &
        (df['Gender'].isin(selected_gender))
    ]

    if filtered_df.empty:
        st.warning("No customers match the selected filters.")
    else:
        st.subheader("Original Data (Filtered)")
        st.dataframe(filtered_df.head())

        if preprocessor and selected_model_name and pca_model:
            X_filtered = filtered_df.drop('CustomerID', axis=1)
            X_processed_filtered = preprocessor.transform(X_filtered)

            model = artifacts.get(selected_model_name.lower())
            if model:
                if selected_model_name in ['DBSCAN', 'AgglomerativeClustering']:
                    clusters = model.fit_predict(X_processed_filtered)
                else:
                    clusters = model.predict(X_processed_filtered)

                filtered_df['Cluster'] = clusters

                st.subheader(f"Cluster Analysis using {selected_model_name}")

                # Evaluation Metrics (only if more than one cluster is found)
                if len(np.unique(clusters)) > 1 and -1 not in np.unique(clusters):
                    try:
                        silhouette = silhouette_score(X_processed_filtered, clusters)
                        davies_bouldin = davies_bouldin_score(X_processed_filtered, clusters)
                        calinski_harabasz = calinski_harabasz_score(X_processed_filtered, clusters)
                        st.write(f"Silhouette Score: {silhouette:.2f}")
                        st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
                        st.write(f"Calinski-Harabasz Score: {calinski_harabasz:.2f}")
                    except Exception as e:
                        st.warning(f"Could not calculate all evaluation metrics: {e}")
                elif -1 in np.unique(clusters) and len(np.unique(clusters)) > 2:
                    # For DBSCAN, if noise points exist but still multiple clusters
                    non_noise_indices = clusters != -1
                    if np.sum(non_noise_indices) > 0 and len(np.unique(clusters[non_noise_indices])) > 1:
                        try:
                            silhouette = silhouette_score(X_processed_filtered[non_noise_indices], clusters[non_noise_indices])
                            davies_bouldin = davies_bouldin_score(X_processed_filtered[non_noise_indices], clusters[non_noise_indices])
                            calinski_harabasz = calinski_harabasz_score(X_processed_filtered[non_noise_indices], clusters[non_noise_indices])
                            st.write(f"Silhouette Score (excluding noise): {silhouette:.2f}")
                            st.write(f"Davies-Bouldin Index (excluding noise): {davies_bouldin:.2f}")
                            st.write(f"Calinski-Harabasz Score (excluding noise): {calinski_harabasz:.2f}")
                        except Exception as e:
                            st.warning(f"Could not calculate all evaluation metrics for DBSCAN: {e}")
                    else:
                        st.info("Not enough non-noise clusters for evaluation metrics.")
                else:
                    st.info("Not enough clusters for evaluation metrics.")

                # PCA for Visualization
                X_pca_filtered = pca_model.transform(X_processed_filtered)
                filtered_df['PCA1'] = X_pca_filtered[:, 0]
                filtered_df['PCA2'] = X_pca_filtered[:, 1]

                # Visualize Clusters
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=filtered_df, palette='viridis', s=100, alpha=0.7, ax=ax)
                ax.set_title(f'{selected_model_name} Clustering (PCA Reduced)')
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
                st.pyplot(fig)

                # Cluster Characteristics
                st.subheader("Cluster Characteristics")
                cluster_summary = filtered_df.groupby('Cluster')[['Age', 'Annual Income', 'Spending Score']].mean()
                st.dataframe(cluster_summary)

                st.subheader("Gender Distribution per Cluster")
                gender_cluster = filtered_df.groupby(['Cluster', 'Gender']).size().unstack(fill_value=0)
                st.dataframe(gender_cluster)

                # Optional: Display Elbow Method and PCA plots if they exist
                st.subheader("Model Training Visualizations")
                try:
                    st.image('models/elbow_method.png', caption='Elbow Method for Optimal K (from training)')
                except FileNotFoundError:
                    st.info("Elbow method plot not found. Run the clustering notebook to generate it.")
                try:
                    st.image('models/kmeans_pca_clusters.png', caption='K-Means PCA Clusters (from training)')
                except FileNotFoundError:
                    st.info("K-Means PCA clusters plot not found. Run the clustering notebook to generate it.")

            else:
                st.warning(f"Model '{selected_model_name}' not loaded. Please ensure the clustering notebook has been run and models are saved.")
        else:
            st.warning("Preprocessing artifacts or PCA model not loaded. Please ensure the clustering notebook has been run.")

else:
    st.info("Please ensure the dataset is in the 'data' directory and the clustering notebook has been run to generate model artifacts.")