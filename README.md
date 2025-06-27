
# 🎬 IMDb Sentiment Predictor

A machine learning-based sentiment analysis app that predicts whether a movie review is **positive** or **negative**.  
The project includes both a **Streamlit web interface** and a **FastAPI backend API**.

---

## 🚀 Live Demo

Access the deployed Streamlit app here:  
🔗 [IMDB Sentiment Predictor App](https://imdb-sentiment-predictor.streamlit.app/)

---

## ✅ Features

- Sentiment prediction for movie reviews
- Web interface using **Streamlit**
- RESTful API using **FastAPI**
- Machine learning model using **TF-IDF** and **Logistic Regression**
- Confidence score for each prediction

---

## 📦 Dataset

Trained on the IMDb Dataset of 50,000 Movie Reviews  
🔗 [Kaggle Dataset Source](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)

---

## 📁 Project Structure

```

imdb-sentiment-predictor/
│
├── train.py               # Training script
├── app.py                 # Streamlit web interface
├── api.py                 # FastAPI service
├── requirements.txt       # Dependencies
│
├── model/
│   ├── sentiment\_model.pkl        # Trained Logistic Regression model
│   └── tfidf\_vectorizer.pkl       # Fitted TF-IDF vectorizer
│
└── data/
└── imdb\_sample.csv            # Subset of IMDb reviews

````

---

## 🧪 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/imdb-sentiment-predictor.git
cd imdb-sentiment-predictor
````

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training the Model

To train the sentiment analysis model:

```bash
python train.py
```

This will:

* Load and clean the IMDb dataset
* Create a TF-IDF vectorizer
* Train a Logistic Regression model
* Save both model and vectorizer into the `model/` directory

---

## 🌐 Web Interface

To run the Streamlit web UI:

```bash
streamlit run app.py
```

You can:

* Enter any movie review
* Get instant sentiment prediction
* View the model's confidence score

---

## 🧠 API Service (Bonus)

To start the FastAPI service:

```bash
uvicorn api:app --reload
```

### 📥 Endpoints:

| Endpoint       | Description                     |
| -------------- | ------------------------------- |
| `GET /`        | Welcome message                 |
| `GET /predict` | Predict sentiment of given text |

**Query Parameter**:

* `text` (string): the movie review

**Example API Request (Python)**

```python
import requests

response = requests.get("http://localhost:8000/predict", params={
    "text": "This movie was fantastic!"
})

print(response.json())
```

---

## 🧩 Programmatic Usage

Use the trained model in your Python code:

```python
import joblib

# Load model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Predict sentiment
text = "Your movie review here"
X = vectorizer.transform([text])
prediction = model.predict(X)

print("Positive" if prediction[0] == 1 else "Negative")
```

---

## ⚙️ Technical Details

| Component          | Description                                        |
| ------------------ | -------------------------------------------------- |
| Text Vectorization | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Model              | Logistic Regression                                |
| Dataset Split      | 80% training / 20% testing                         |
| Preprocessing      | English stop-word removal                          |

---

## 📚 Dependencies

* `scikit-learn`: ML functionality
* `pandas`: Data loading/manipulation
* `streamlit`: UI framework
* `fastapi`: API framework
* `joblib`: Save/load ML models
* `uvicorn`: FastAPI ASGI server

---

## 🤝 Contributing

Pull requests, suggestions, and improvements are welcome.
Feel free to open an issue or submit a PR!

---

## 📄 License

MIT License — you are free to use, modify, and distribute this project.

---

## 👨‍💻 Author

Made with ❤️ by [Nebiyou Elias](https://github.com/resourceful-nebil)

```


