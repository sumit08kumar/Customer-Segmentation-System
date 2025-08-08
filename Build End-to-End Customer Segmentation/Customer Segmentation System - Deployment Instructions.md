# Customer Segmentation System - Deployment Instructions

## Docker Deployment

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, for easier management)

### Option 1: Using Docker Compose (Recommended)

1. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

2. **Run in detached mode:**
   ```bash
   docker-compose up -d --build
   ```

3. **Stop the application:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker directly

1. **Build the Docker image:**
   ```bash
   docker build -t customer-segmentation .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 customer-segmentation
   ```

3. **Run in detached mode:**
   ```bash
   docker run -d -p 8501:8501 --name customer-segmentation-app customer-segmentation
   ```

4. **Stop the container:**
   ```bash
   docker stop customer-segmentation-app
   docker rm customer-segmentation-app
   ```

### Accessing the Application

Once the container is running, open your web browser and navigate to:
```
http://localhost:8501
```

### Features Available

- **Interactive Clustering Models**: K-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models
- **Real-time Filtering**: Filter customers by age, income, spending score, and gender
- **Visualization**: PCA-reduced cluster plots with interactive legends
- **Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
- **Cluster Analysis**: View cluster characteristics and gender distribution

### Troubleshooting

1. **Port already in use:**
   ```bash
   # Use a different port
   docker run -p 8502:8501 customer-segmentation
   ```

2. **Check container logs:**
   ```bash
   docker logs customer-segmentation-app
   ```

3. **Health check:**
   ```bash
   curl http://localhost:8501/_stcore/health
   ```

### Cloud Deployment

This Docker container can be deployed to any cloud platform that supports Docker:

- **AWS**: ECS, Fargate, or EC2
- **Google Cloud**: Cloud Run, GKE, or Compute Engine
- **Azure**: Container Instances, AKS, or Virtual Machines
- **Heroku**: Container Registry
- **DigitalOcean**: App Platform or Droplets

### Environment Variables

The application supports the following environment variables:

- `STREAMLIT_SERVER_PORT`: Port for the Streamlit server (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Address to bind the server (default: 0.0.0.0)

### Data and Models

The application includes:
- Sample customer dataset (200 customers)
- Pre-trained clustering models
- PCA transformation for visualization
- Preprocessing pipeline

All models and data are included in the Docker image for immediate use.

