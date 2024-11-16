"""
Script Name: predict.py
Description: Implements the /predict endpoint to handle housing price predictions. This script includes input validation, 
             feature engineering, and model prediction logic for both clustering and regression models.
Version: 1.0.0
Author: Dimitris Tagkoulis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ValidationError
import pandas as pd
from app.scripts.preprocessing import feature_engineering
from app.scripts.modeling_pipeline import load_model
import logging
import json
from pathlib import Path


class PredictionRequest(BaseModel):
    features: dict


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

router = APIRouter()

from app.config.base import BASE_DIR

# Use BASE_DIR to construct absolute paths
clustering_dir = BASE_DIR / "models" / "clustering"
regression_dir = BASE_DIR / "models" / "regression"


def load_feature_names(model_path):
    # Ensure model_path is a Path object
    model_path = Path(model_path)

    # Construct the feature names path by appending "_features.json"
    feature_names_path = model_path.parent / f"{model_path.stem}_features.json"

    if not feature_names_path.exists():
        raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")

    # Load the feature names from the JSON file
    with open(feature_names_path, "r") as f:
        return json.load(f)


# Function to check data types of incoming features
def check_data_types(features):
    # Define expected types for each feature
    expected_types = {
        "longitude": (float, int),  # Allow both float and int
        "latitude": (float, int),  # Allow both float and int
        "housing_median_age": (float, int),
        "total_rooms": (float, int),
        "total_bedrooms": (float, int),
        "population": (float, int),
        "households": (float, int),
        "median_income": (float, int),
        "median_house_value": (float, int),
        "ocean_proximity": str,
    }

    # Check if the types match
    for feature, expected_types_tuple in expected_types.items():
        if feature not in features:
            raise HTTPException(status_code=400, detail=f"Missing feature: {feature}")

        feature_value = features[feature]

        # If the feature is numeric (int or float), attempt to convert to float
        if isinstance(feature_value, (int, float)):
            features[feature] = float(feature_value)
        elif not isinstance(feature_value, expected_types_tuple):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type for {feature}. Expected one of {expected_types_tuple}, got {type(feature_value).__name__}.",
            )

    return True


@router.post("/predict")
async def predict(request: PredictionRequest):
    logger.info("Received prediction request")

    # Extract features from request
    features = request.dict()["features"]
    logger.debug(f"Features extracted: {features}")

    try:
        # Validate data types
        check_data_types(features)
        logger.info("Data types validated successfully")

        # Load models
        logger.info("Loading regression and clustering models")
        model = load_model(regression_dir)
        kmeans = load_model(clustering_dir)

        # Load feature names used during training
        feature_names = load_feature_names(clustering_dir / "kmeans_model.pkl")
        logger.debug(f"Feature names loaded: {feature_names}")

        if model is None or kmeans is None:
            logger.error("One or both models failed to load.")
            raise HTTPException(status_code=500, detail="Failed to load models.")

        logger.info("Models loaded successfully")

        # Convert input features to a DataFrame for feature engineering
        input_data = pd.DataFrame([features])
        logger.debug(f"Input data for feature engineering: {input_data}")

        # Apply feature engineering
        logger.info("Applying feature engineering")
        input_data = feature_engineering(input_data)
        logger.debug(f"Feature engineering applied: {input_data}")

        # Ensure that necessary columns are present after feature engineering
        X = input_data.drop("median_house_value", axis=1, errors="ignore")

        # Align input data with the saved feature names
        for missing_feature in set(feature_names) - set(X.columns):
            X[missing_feature] = 0
        X = X[feature_names]

        # Check for missing values in the features (not the clusters) before applying KMeans
        if X.isnull().sum().any():
            logger.error("Missing values found in input data")
            raise HTTPException(status_code=400, detail="Missing values in input data")
        logger.info("No missing values in the input data")

        # Add cluster column using KMeans model
        logger.info("Predicting clusters using KMeans model")
        cluster_predictions = kmeans.predict(X)
        X["cluster"] = cluster_predictions
        logger.debug(f"Cluster predictions: {cluster_predictions}")

        # Make prediction using regression model
        logger.info("Making prediction using regression model")
        prediction = model.predict(X)
        logger.debug(f"Prediction result: {prediction}")

        # Return the prediction result
        return {"predicted_price": prediction[0], "cluster": int(X["cluster"][0])}

    except ValidationError as ve:
        logger.error(f"Validation error: {ve.errors()}")
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except KeyError as e:
        logger.error(f"Invalid feature: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid feature: {str(e)}")
    except Exception as e:
        logger.exception("An unexpected error occurred")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
