# app.py
import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Streamlit UI
st.set_page_config(page_title="IMDb Sentiment Analyzer", layout="centered")

st.title("🎬 IMDb Sentiment Analyzer")
st.write("Enter a movie review below to see if it's positive or negative.")

review = st.text_area("Movie Review", placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        X = vectorizer.transform([review])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        label = "Positive 😊" if pred == 1 else "Negative 😞"
        confidence = round(np.max(proba) * 100, 2)

        st.markdown(f"### Prediction: {label}")
        st.markdown(f"**Confidence:** {confidence}%")
