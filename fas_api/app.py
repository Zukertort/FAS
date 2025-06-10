# FastAPI application logic will go here
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager

# --- Pydantic Models for Request and Response ---
class ReviewRequest(BaseModel):
    """Pydantic model for the input text."""
    text: str

class ReviewResponse(BaseModel):
    """Pydantic model for the API response."""
    text: str
    sentiment: str
    confidence: float

# --- Load Model and Vectorizer at Startup ---
model = None
vectorizer = None

@asynccontextmanager
async def load_model_artifacts():
    """
    Load the trained model and vectorizer from disk when the app starts.
    """
    global model, vectorizer
    model_path = os.path.join('saved_model', 'model.joblib')
    vectorizer_path = os.path.join('saved_model', 'vectorizer.joblib')

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        print("--- Model and vectorizer loaded successfully ---")
    except FileNotFoundError:
        print(f"ERROR: Model or vectorizer not found at specified paths.")
        print("Please run the 'train_model.py' script to generate them.")
        model = None
        vectorizer = None

# Define the FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to predict the sentiment of a movie review using a pre-trained Naive Bayes model.",
    version="1.0.0"
)

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
def read_root():
    """Root endpoint to check if the API is running."""
    return {"status": "ok", "message": "Sentiment Analysis API is running."}

@app.post("/predict", response_model=ReviewResponse, tags=["Prediction"])
def predict_sentiment(request: ReviewRequest):
    """
    Receives a text string, processes it, and returns the sentiment prediction.
    """
    if not model or not vectorizer:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model and restart the server."
        )
    
    # 1. Extract text from the request
    input_text = request.text

    # 2. Vectorize the input text using the loaded vectorizer
    vectorized_text = vectorizer.transform([input_text])

    # 3. Make a prediction using the loaded model
    prediction = model.predict(vectorized_text)[0]

    # 4. Get the prediction probabilities for confidence score
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = float(np.max(probabilities))

    # 5. Return the structured response
    return ReviewResponse(
        text=input_text,
        sentiment=prediction,
        confidence=confidence
    )

# This allows running the app directly using `python app.py`
if __name__ == "__app__":
    uvicorn.run(app, host="127.0.0.1", port=8000)