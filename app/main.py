# main.py
from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.predict import router as predict_router

app = FastAPI()

# Include the routes
app.include_router(health_router)
app.include_router(predict_router)


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML FastAPI application"}


"""
Script Name: main.py
Description: Entry point for the FastAPI application. Configures and starts the application, including routing for 
             health checks and predictions.
Version: 1.0.0
Author: Dimitris Tagkoulis
"""

from fastapi import FastAPI
from app.api.health import router as health_router
from app.api.predict import router as predict_router

# Initialize the FastAPI application
app = FastAPI()

# Include the API routes
app.include_router(health_router)
app.include_router(predict_router)


@app.get("/")
def read_root():
    """
    Root endpoint to provide a welcome message.

    Returns:
    - dict: Welcome message.
    """
    return {"message": "Welcome to the ML FastAPI application"}
