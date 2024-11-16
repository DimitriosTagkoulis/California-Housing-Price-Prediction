import joblib
import pandas as pd
import numpy as np
import logging
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Path to the model file (adjust to your actual path)
model_dir = "../models/regression/"


# Dummy test input data (replace with actual features used during training)
test_input_data = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",  # Adjust this to match the training data format
}


def load_model(model_dir):
    """
    Function to load the model using joblib.
    """
    try:
        logger.info(f"Loading model from {model_dir}")

        # Check if the directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"The directory {model_dir} does not exist.")

        # List all the files in the directory
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}.")

        # Get the full path of the most recently modified model file
        latest_model_file = max(
            model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f))
        )

        latest_model_path = os.path.join(model_dir, latest_model_file)

        # Load the model from the latest file
        model = joblib.load(latest_model_path)

        if hasattr(model, "booster_"):
            logger.info("Model loaded successfully.")
        else:
            raise ValueError(
                "Model does not have 'booster_' attribute, might not be a valid LightGBM model."
            )
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def preprocess_input_data(input_data):
    """
    Function to preprocess input data (feature engineering steps as required).
    """
    # Convert to DataFrame for feature engineering and ensure correct format
    input_df = pd.DataFrame([input_data])
    logger.info(f"Input DataFrame created:\n{input_df}")

    # Apply feature engineering steps (add your custom preprocessing here)
    # For simplicity, I will just scale the numerical columns for this example.
    numerical_columns = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    scaler = StandardScaler()
    input_df[numerical_columns] = scaler.fit_transform(input_df[numerical_columns])
    logger.info(f"Scaled input DataFrame:\n{input_df}")

    return input_df


def predict_with_model(model, input_data):
    """
    Function to make predictions using the trained LightGBM model.
    """
    try:
        # Ensure input data has the same columns/features as the model expects
        logger.info("Making prediction...")
        prediction = model.predict(input_data)
        logger.info(f"Prediction result: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise


def test_lightgbm_model():
    """
    Main function to test the LightGBM model loading and prediction.
    """
    try:
        # Step 1: Load the model
        model = load_model(model_dir)

        # Step 2: Preprocess the input data (feature engineering)
        processed_data = preprocess_input_data(test_input_data)

        # Step 3: Make the prediction
        prediction = predict_with_model(model, processed_data)

        # Step 4: Output the result
        print(f"Predicted Value: {prediction[0]}")
    except Exception as e:
        logger.error(f"An error occurred during testing: {str(e)}")


if __name__ == "__main__":
    test_lightgbm_model()
