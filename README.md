California Housing Price Prediction
This project predicts housing prices based on a dataset of California housing features. It includes a FastAPI application that serves predictions using a machine learning model. The project is organized into modular scripts for data preprocessing, feature engineering, and model training. The FastAPI application serves the model via a REST API and provides a health check endpoint.

Table of Contents
Overview
Installation
Usage
API Endpoints
Predict
Health Check
File Structure
Modeling Details
License
Overview
The objective of this project is to create a machine learning model that predicts the housing price of a property based on a variety of features such as geographic location, housing age, and income level. The model is trained using the California Housing Prices dataset.

The FastAPI application exposes two main endpoints:

/predict for getting price predictions.
/health for checking the health of the API.
The project follows modular and scalable architecture, utilizing error handling, logging, and model versioning.

Installation
1. Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/california-housing-price-prediction.git
cd california-housing-price-prediction
2. Create a virtual environment (optional but recommended):
bash
Copy code
python -m venv myenv
source myenv/bin/activate
# On Windows, use `myenv\Scripts\activate`
3. Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
1. Start the FastAPI application:
Run the FastAPI server locally to access the prediction API.

bash
Copy code
uvicorn app.main:app --reload
2. API Endpoints:
Predict: /predict
Method: POST
Description: Accepts a JSON payload with features and returns the predicted housing price.
Request body (JSON):

json
Copy code
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
Response:

json
Copy code
{
  "predicted_price": 168000.0,
  "cluster": 2
}
Health Check: /health
Method: GET
Description: Returns the health status of the API. It checks whether the models are loaded properly.
Response:

json
Copy code
{
  "status": "healthy"
}
If there's an error loading the model, the status will be "unhealthy" with the error details.

File Structure
graphql
Copy code
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
Modeling Details
Data Preprocessing
Missing Values: Rows with missing values are dropped.
Feature Scaling: Numerical features are scaled.
Encoding: The categorical variable ocean_proximity is one-hot encoded.
Feature Engineering
Clustering: Neighborhood effects are captured by adding a "cluster" feature using KMeans clustering.
Feature Transformation: Includes feature interactions, log transformations, and geospatial features (e.g., distance from the ocean).
Models
Random Forest: Robust to overfitting and handles high-dimensional data.
XGBoost: Optimized gradient boosting.
CatBoost: Efficient gradient boosting for categorical features.
Metrics for Evaluation:

RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)
MASE (Mean Absolute Scaled Error)
Model Versioning
Best models are saved in the models/ directory with a timestamp.
License
This project is licensed under the MIT License.

Feel free to replace the placeholder GitHub link and any other specifics as needed.