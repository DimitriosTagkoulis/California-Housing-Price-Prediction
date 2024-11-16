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

# Send POST request to the API
response = requests.post(url, json=features)

# Check if the request was successful
if response.status_code == 200:
    prediction = response.json()
    print(f"Predicted house price: {prediction['predicted_price']}")
    print(f"Cluster: {prediction['cluster']}")
else:
    print(f"Error: {response.status_code}, {response.text}")
