# api.py
from fastapi import FastAPI, Query
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Create app
app = FastAPI(
    title="IMDb Sentiment API",
    description="A simple API to predict sentiment of movie reviews",
    version="1.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to the IMDb Sentiment Analysis API!"}

@app.get("/predict")
def predict(text: str = Query(..., min_length=3, description="Movie review text")):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = "positive" if pred == 1 else "negative"
    confidence = round(np.max(proba) * 100, 2)

    return {
        "prediction": label,
        "confidence": f"{confidence}%"
    }
