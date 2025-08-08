# Customer-Segmentation-System

Great choice, Sumit! Unsupervised learning is often underrepresented on resumes, so a well-executed end-to-end **unsupervised ML project** with model comparison + deployment will make your portfolio **stand out**.

---

## 📌 Project Title:

**Customer Segmentation System: An End-to-End Unsupervised Machine Learning Pipeline with Web App Deployment**

---

## 🧠 Project Summary:

This project implements an end-to-end unsupervised machine learning system to segment customers based on behavioral and demographic data. It showcases multiple clustering algorithms, dimensionality reduction techniques, and an interactive Streamlit app for dynamic cluster analysis — all containerized and ready for cloud deployment.

---

## 🎯 Problem Statement:

Businesses often struggle to understand their diverse customer base. The goal is to **group customers into similar segments** so that companies can:

* Personalize marketing
* Design targeted promotions
* Improve customer retention strategies

---

## 📊 Dataset:

* **Source**: [Kaggle – Mall Customer Segmentation Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial)
* **Features**:

  * `CustomerID`
  * `Gender`
  * `Age`
  * `Annual Income`
  * `Spending Score`

> You can also upgrade this with e-commerce or telecom customer datasets (optional).

---

## ✅ Key Steps:

### 1. 🔍 Exploratory Data Analysis (EDA)

* Histograms, pairplots, violin plots (age, income, gender)
* Correlation matrix to identify informative features

---

### 2. 🧼 Preprocessing

* One-hot encoding for `Gender`
* Standardization of numerical features
* Dimensionality reduction with **PCA** or **t-SNE** for visualization

---

### 3. 🧠 Unsupervised Model Comparison

| Algorithm                                   | Use Case                                    |
| ------------------------------------------- | ------------------------------------------- |
| **K-Means**                                 | Baseline hard clustering                    |
| **Hierarchical Clustering (Agglomerative)** | Tree-based segmentation                     |
| **DBSCAN**                                  | Density-based detection (good for outliers) |
| **Gaussian Mixture Models (GMM)**           | Probabilistic clustering                    |
| (Optional) **Autoencoders**                 | Deep clustering for nonlinear patterns      |

---

### 4. 📈 Evaluation Metrics

* Silhouette Score
* Davies–Bouldin Index
* Calinski–Harabasz Score
* Elbow method & Dendrogram for cluster tuning

---

### 5. 📊 Visualization

* 2D PCA/t-SNE cluster plots
* Interactive **Streamlit UI** with cluster count selector and segment interpretation

---

## 🌐 Deployment:

### ✅ Streamlit Web App:

* Input sliders to filter and explore different segments
* Display cluster characteristics (mean income, spending habits)
* Visualize cluster plots and statistics

### ✅ Dockerized Deployment:

* Dockerfile to containerize the app
* Optional deployment to:

  * **Streamlit Cloud**
  * **Render**
  * **GCP Cloud Run**
  * **AWS EC2**

---

## 📁 Folder Structure:

```
customer-segmentation/
├── app/
│   └── streamlit_app.py
├── data/
│   └── mall_customers.csv
├── models/
│   └── kmeans.pkl
│   └── scaler.pkl
├── notebooks/
│   └── EDA.ipynb
│   └── Clustering_Models.ipynb
├── Dockerfile
├── requirements.txt
├── README.md
```

---

## ✨ Resume-Ready Bullet Point:

> ✅ Built and deployed an end-to-end unsupervised machine learning system for customer segmentation using K-Means, DBSCAN, Hierarchical Clustering, and GMM. Evaluated clustering quality with silhouette and Davies–Bouldin scores. Deployed an interactive Streamlit app with Docker to explore customer segments dynamically.

---

## Bonus Ideas (Optional Add-ons):

* Add **SHAP for GMM soft clusters**
* Use **Elbow + Silhouette** combined cluster optimization
* Combine with supervised models (semi-supervised learning)
