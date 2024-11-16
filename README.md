# California Housing Price Prediction

This project predicts housing prices based on a dataset of California housing features ([source](https://www.kaggle.com/datasets/camnugent/california-housing-prices)).
It includes a FastAPI application that serves predictions using a machine learning model.
The project is organized into modular scripts for data preprocessing, feature engineering, and model training.
The FastAPI application serves the model via a REST API and provides a prediction and a health check endpoint.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
    - [Predict](#predict)
    - [Health Check](#health-check)
- [File Structure](#file-structure)
- [Modeling Details](#modeling-details)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Models](#models)
  - [Evaluation Metrics](#Evaluation-Metrics)
- [License](#license)

## Overview

The objective of this project is to create a machine learning model that predicts the housing price of a property based on a variety of features such as geographic location, housing age, and income level. The model is trained using the California Housing Prices dataset.

The FastAPI application exposes two main endpoints:

- `/predict` for getting price predictions.
- `/health` for checking the health of the API.

The project follows modular and scalable architecture, utilizing error handling, logging, and model versioning.

---

## Installation

1. **Clone the repository:**

 ```
   git clone https://github.com/yourusername/california-housing-price-prediction.git
   cd california-housing-price-prediction
```

2. **Create a virtual environment (optional but recommended):**

```python
python -m venv myenv
source myenv/bin/activate
# On Windows, use `myenv\Scripts\activate`
```

3. **Install the required dependencies:**

```python
pip install -r requirements.txt
```

---

## Usage

### 1. Start the FastAPI application

Run the FastAPI server locally to access the prediction API.

```python
uvicorn app.main:app --reload
```

### 2. API Endpoints

#### Predict: `/predict`

- **Method:** `POST`
- **Description:** Accepts a JSON payload with features and returns the predicted housing price.

**Request Body (JSON):**

```json
{
  "features": {
    "longitude": -118.25,
    "latitude": 34.05,
    "housing_median_age": 41.0,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 5.0,
    "ocean_proximity": "NEAR BAY",
    "median_house_value": 0
  }
}
```

**Response:**

```json
{
  "predicted_price": 168000.0,
  "cluster": 2
}
```

#### Health Check: `/health`

- **Method:** `GET`
- **Description:** Returns the health status of the API. It checks whether the models are loaded properly.

**Response:**

```json
{
 "status": "healthy"
}
```

If there's an error loading the model, the status will be "unhealthy" with the error details.

## File Structure

```
california-housing-price-prediction/
│
├── app/
│   ├── api/                        # API routes for FastAPI
│   │   ├── health.py               # Health check endpoint
│   │   └── predict.py              # Prediction endpoint
│   ├── config/                     # Configuration files
│   │   └── base.py                 # Base configuration for paths
│   ├── scripts/                    # Scripts for modeling and preprocessing
│   │   ├── EDA.py                  # Exploratory data analysis
│   │   ├── model_initialization.py # Main script for training models
│   │   ├── modeling_pipeline.py    # Training, evaluation, and saving models
│   │   └── preprocessing.py        # Data preprocessing and feature engineering
│   ├── utils/                      # Utility functions
│   ├── main.py                     # Entry point for FastAPI
│   └── __init__.py                 # Package initialization
│
├── data/
│   └── housing.csv                 # California housing dataset
│
├── models/
│   ├── clustering/
│   │   ├── kmeans_model.pkl        # KMeans clustering model
│   │   └── kmeans_model_features.json # Features used in clustering
│   ├── regression/
│   │   └── Random Forest_xxxxx.pkl # Saved regression model
│
├── logs/                           # Logs for debugging and tracking
│   ├── catboost_info/              # CatBoost logs
│
├── mlruns/                         # MLflow logs
│
├── myenv/                          # Virtual environment
│
├── .gitignore                      # Git ignore rules
├── README.md                       # Documentation
└── requirements.txt                # Python dependencies

```

## Modeling Details

### Data Preprocessing

- **Missing Values:** Rows with missing values are dropped.
- **Feature Scaling:** Numerical features are scaled.
- **Encoding:** The categorical variable `ocean_proximity` is one-hot encoded.

### Feature Engineering

- **Clustering:** Neighborhood effects are captured by adding a "cluster" feature using KMeans clustering.
- **Feature Transformation:** Includes feature interactions, log transformations, and geospatial features (e.g., distance from the ocean).

### Models

- **Random Forest:** Robust to overfitting and handles high-dimensional data.
- **XGBoost:** Optimized gradient boosting.
- **CatBoost:** Efficient gradient boosting for categorical features.

### Evaluation Metrics

##### 1. Root Mean Squared Error (RMSE)

- **Why Chosen?**
  - RMSE penalizes large errors more than smaller ones because it squares the errors. This is particularly useful when predicting house prices, where large deviations are more impactful (e.g., underpricing a house by 100,000 is a more critical error than by 1,000).
  - It provides results in the same units as the target variable (e.g., dollars), making it easy to interpret.
- **Strengths:**
  - Sensitive to outliers, which is useful in datasets with significant price variations.
- **Weaknesses:**
  - Heavily affected by outliers, which can sometimes overly bias the metric.
- **Justification:**
  - RMSE is a standard metric in regression problems and provides a good measure of how well the model is capturing the variability in house prices.

##### 2. Mean Absolute Percentage Error (MAPE)

- **Why Chosen?**
  - MAPE calculates the average percentage error, making it scale-independent and useful for comparing models across datasets with varying price ranges.
  - It provides an intuitive understanding of the model's accuracy by showing errors as percentages.
- **Strengths:**
  - Easy to interpret for stakeholders since it expresses errors in relative terms.
- **Weaknesses:**
  - Can be skewed when actual values are close to zero, as the percentage error becomes disproportionately large.
- **Justification:**
  - MAPE is valuable for understanding how close predictions are to actual values in relative terms, especially when different house prices span multiple orders of magnitude.

##### 3. Mean Absolute Scaled Error (MASE)

- **Why Chosen?**
  - MASE is designed to compare forecast accuracy across datasets and time series models. It scales the error based on a baseline (e.g., naive forecasting).
  - It is particularly robust to different scales and units, making it a versatile metric.
- **Strengths:**
  - Not overly sensitive to outliers.
  - Provides a baseline comparison, offering insights into whether the model outperforms a naive forecast.
- **Weaknesses:**
  - Less commonly used in some domains, which might make interpretation less familiar to non-technical stakeholders.
- **Justification:**
  - MASE adds another layer of insight by contextualizing the model’s performance relative to a naive approach.

---

#### Final Metric Choice

While all three metrics are valuable for understanding different aspects of model performance, **RMSE** is chosen as the final evaluation metric. This decision is based on the following reasoning:

- RMSE directly aligns with the project's primary goal of minimizing large pricing errors, which are particularly impactful in the context of housing price prediction.
- It is interpretable in the same units as the target variable, making it more meaningful to stakeholders.
- While MAPE and MASE provide additional insights, they are complementary metrics rather than the primary ones for this context. For instance:
  - MAPE is useful for understanding percentage-based accuracy.
  - MASE helps evaluate performance relative to a naive model.
- RMSE balances sensitivity to large deviations (important for high-stakes predictions like housing prices) with interpretability, making it the most appropriate choice.


The final metric for evaluating the model's performance is **RMSE**, supported by **MAPE** and **MASE** as complementary metrics for additional context and validation. This combination ensures a well-rounded assessment of the model's accuracy and robustness.

---

### Model Versioning

Best models are saved in the `models/` directory with a timestamp.

## Future Work

To further improve and expand the California Housing Price Prediction project, the following enhancements are suggested:

1. **Implementation of a Training Endpoint:**
   - Develop a `/train` endpoint in the FastAPI application to automate the training process on new data.
   - Allow users to upload new datasets through the endpoint, triggering preprocessing, feature engineering, and model retraining.
   - Incorporate model evaluation and versioning to ensure that only better-performing models are deployed.
   - Add safeguards to validate the quality of the new dataset (e.g., schema validation, missing value checks) before training.
2. **Advanced Feature Engineering:**
   - **Geospatial Features:**
     - Incorporate geospatial features such as proximity to schools, parks, hospitals, and public transportation.
     - Use geocoding APIs to generate features like distance from landmarks or city centers.
   - **Temporal Features:**
     - Add temporal features such as seasonality effects, year-over-year trends, or market cycles in housing prices.
     - Include derived features like the age of the property or time since last renovation.
   - **Feature Interactions:**
     - Explore interactions between features such as income level and population density or housing age and total rooms.
     - Use polynomial feature transformations to capture non-linear relationships.
   - **Feature Selection Techniques:**
     - Use statistical methods such as ANOVA, mutual information, or chi-square tests to identify the most relevant features.
     - Implement tree-based feature importance from models like Random Forest or XGBoost to rank and select top features.
     - Apply recursive feature elimination (RFE) or principal component analysis (PCA) to reduce dimensionality and eliminate irrelevant or redundant features.
     - Explore L1 regularization (Lasso) for sparse feature selection in linear models.
   - **External Data Sources:**
     - Enrich the dataset by integrating external data sources, such as economic indicators (e.g., unemployment rates, inflation) or environmental factors (e.g., air quality, crime rates).
   - **Custom Features:**
     - Develop new features such as housing density (total_rooms/population) or income-per-household (median_income/households) for more domain-specific insights.
3. **Model Optimization:**
   - Experiment with ensemble learning techniques such as stacking and blending for better performance.
   - Fine-tune hyperparameters using automated tools like Optuna or Hyperopt.

4. **Additional Models:**
   - Integrate neural networks (e.g., deep learning models) for complex feature interactions.
   - Explore transfer learning with pre-trained models for geospatial or tabular data.
5. **Improved API Features:**
   - Add support for batch predictions to handle multiple data points in a single request.
   - Implement input validation and schema generation using tools like Pydantic.

6. **Visualization:**
   - Create interactive dashboards for end users to visualize prediction insights.
   - Provide explanations for predictions using SHAP or LIME for model interpretability.

7. **Scalability:**
   - **Cloud Deployment:**
     - Deploy the application on cloud platforms such as AWS, Google Cloud Platform (GCP), or Azure to ensure high availability and fault tolerance.
     - Use managed services like AWS Elastic Beanstalk, Google App Engine, or Azure App Service for easier deployment and scaling.
   - **Containerization:**
     - Dockerize the application to create portable and consistent environments for development, testing, and production.
     - Use Docker Compose to manage dependencies and multi-container setups during local development.
   - **Orchestration with Kubernetes:**
     - Deploy the application using Kubernetes for advanced orchestration, including automated scaling, self-healing, and zero-downtime deployments.
     - Use tools like Helm to manage and version Kubernetes configurations efficiently.
   - **Load Balancing:**
     - Use cloud-native load balancers (e.g., AWS Application Load Balancer or GCP Load Balancer) to distribute traffic evenly across instances.
     - Enable health checks to route requests only to healthy application instances.
   - **Horizontal Scaling:**
     - Configure auto-scaling groups to dynamically adjust the number of instances based on traffic load.
     - Use metrics like CPU utilization, memory usage, and request latency to trigger scaling events.
   - **Database Scaling:**
     - Use a managed database service like AWS RDS, Google Cloud SQL, or Azure SQL Database with automatic scaling and backups.
     - Implement read replicas for scaling read-heavy workloads.
     - Optimize database queries and use indexing to improve performance under high loads.
   - **Caching:**
     - Integrate caching mechanisms like Redis or Memcached to reduce database load and improve API response times.
     - Cache frequently requested data, such as clustering results or static metadata, at the application or database level.
   - **Content Delivery Network (CDN):**
     - Use a CDN like Cloudflare or AWS CloudFront to serve static assets, reducing server load and improving response times for global users.
   - **Monitoring and Alerts:**
     - Set up robust monitoring tools like Prometheus, Grafana, or Datadog to track system performance and detect bottlenecks.
     - Implement alerts for high traffic loads, slow response times, or server failures to enable proactive troubleshooting.

9. **Integration with External Systems:**
    - Connect the API to real estate platforms or CRMs for seamless integration.
    - Enable export of predictions and insights into commonly used file formats or systems.

10. **Enhanced Data Handling:**
    - Address missing data using imputation techniques instead of row dropping.
    - Incorporate advanced data augmentation strategies for rare cases in the dataset.

11. **Improved Logging:**
    - Enhance logging with structured formats such as JSON to improve readability and integration with log management tools.
    - Use centralized logging systems like ELK Stack (Elasticsearch, Logstash, Kibana) or cloud-based solutions (e.g., AWS CloudWatch or Google Cloud Logging).
    - Add logging for key metrics, API performance, and model predictions to monitor the system’s health and efficiency.

12. **CI/CD:**
    - Implement a continuous integration/continuous delivery (CI/CD) pipeline to automate testing, building, and deployment processes.
    - Use tools like GitHub Actions, Jenkins, or GitLab CI/CD to ensure seamless integration and deployment.
    - Integrate automated unit tests, regression tests, and performance tests to maintain code quality.
    - Enable deployment strategies like blue-green or canary deployments to minimize downtime during updates.
    - Add version control for models and data pipelines to track and manage changes effectively.

## License

This project is licensed under the MIT License.
