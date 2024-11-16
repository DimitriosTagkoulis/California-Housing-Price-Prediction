"""
Script Name: api_client.py
Description: Sample Script that sends a POST request to the FastAPI prediction endpoint with housing features and displays the predicted
             house price and cluster information.
Version: 1.0.0
Author: Dimitris Tagkoulis
"""

import requests

# URL of the FastAPI prediction endpoint
url = "http://127.0.0.1:8000/predict"

# Features to be sent in the POST request
features = {
    "features": {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
        "median_house_value": 0,  # Placeholder, since it's not used for prediction
    }
}


def send_prediction_request(url, features):
    """
    Sends a POST request to the FastAPI prediction endpoint with given features.

    Parameters:
    - url (str): URL of the prediction endpoint.
    - features (dict): Dictionary containing input features for prediction.

    Returns:
    - dict: Prediction results if successful.
    - None: If the request fails.
    """
    try:
        response = requests.post(url, json=features)

        if response.status_code == 200:
            prediction = response.json()
            print(f"Predicted house price: {prediction['predicted_price']}")
            print(f"Cluster: {prediction['cluster']}")
            return prediction
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None


if __name__ == "__main__":
    send_prediction_request(url, features)
