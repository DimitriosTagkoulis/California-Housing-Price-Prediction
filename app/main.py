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
