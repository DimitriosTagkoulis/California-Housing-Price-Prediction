from fastapi import APIRouter, HTTPException
import os
from app.scripts.modeling_pipeline import load_model

# Initialize the health check API router
router = APIRouter()

# Define model directories
clustering_dir = "./models/clustering/"
regression_dir = "./models/regression/"


def is_directory_empty(directory_path):
    """
    Check if the specified directory is empty.

    Parameters:
    - directory_path (str): Path to the directory.

    Returns:
    - bool: True if directory is empty, False otherwise.
    """
    return not bool(os.listdir(directory_path))


# Health check endpoint
@router.get("/health")
async def health_check():
    try:
        # Check if the model directories are not empty and load the models
        if is_directory_empty(regression_dir):
            raise HTTPException(
                status_code=400, detail="Regression model directory is empty."
            )
        if is_directory_empty(clustering_dir):
            raise HTTPException(
                status_code=400, detail="Clustering model directory is empty."
            )

        # Attempt to load models
        load_model(regression_dir)
        load_model(clustering_dir)

        return {"status": "healthy"}
    except HTTPException as he:
        # If there's an error with loading the models, return an error response
        return {"status": "unhealthy", "detail": he.detail}
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}
