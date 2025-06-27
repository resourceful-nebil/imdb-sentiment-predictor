# 🎬 IMDb Sentiment Analyzer

A simple and interactive web app that predicts the sentiment of IMDb movie reviews — whether they’re **positive** or **negative** — using classical machine learning.

Built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 🧠 How It Works

- The model is trained on a subset of IMDb movie reviews.
- It uses **TF-IDF vectorization** and a **Logistic Regression** classifier.
- Users can input a movie review in the UI and get a **prediction** with a **confidence score**.

---

## 📁 Project Structure

```

imdb-sentiment-predictor/
│
├── app.py                  # Streamlit UI for predictions
├── train.py                # Script to train and save the model
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
│
├── data/
│   └── imdb\_sample.csv     # Subset of IMDb reviews (\~5000 rows)
│
└── model/
├── sentiment\_model.pkl         # Trained Logistic Regression model
└── tfidf\_vectorizer.pkl        # Saved TF-IDF vectorizer

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/imdb-sentiment-predictor.git
cd imdb-sentiment-predictor
````

### 2. Install Dependencies

Make sure you have Python 3.7+ installed, then run:

```bash
pip install -r requirements.txt
```

### 3. (Optional) Train the Model

If you want to train from scratch:

```bash
python train.py
```

This will:

* Load the data from `data/imdb_sample.csv`
* Train the classifier
* Save the model and vectorizer in the `model/` directory

---

## 💻 Run the Web App

```bash
streamlit run app.py
```

Then open the local URL provided in your terminal (usually [http://localhost:8501](http://localhost:8501)).

---

## 📝 Example

Input:

```
This movie was absolutely incredible. I loved every second of it!
```

Output:

```
Prediction: Positive 😊
Confidence: 93.75%
```

---

## 📦 Dependencies

* `scikit-learn`
* `pandas`
* `numpy`
* `joblib`
* `streamlit`

All are listed in `requirements.txt`.

---

## 🌍 Deployment

You can deploy this project on [Streamlit Cloud](https://share.streamlit.io/) in minutes.

Just push your repo to GitHub and connect it with Streamlit Cloud to go live.

---

## 📜 License

MIT License — feel free to use, modify, and share.



