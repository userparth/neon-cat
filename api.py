import numpy as np
import pickle
import os
import logging
import subprocess
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File Paths
MODEL_PATH = "models/svm_sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Check if Model Exists, If Not, Train It
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    logging.warning("Model and vectorizer files not found. Training the models automatically...")
    subprocess.run(["python", "train.py"], check=True)

# Load Model and Vectorizer After Training
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Model and vectorizer loaded successfully.")
else:
    logging.error("Model training failed. Please check train.py for errors.")
    model, vectorizer = None, None

# Initialize Flask App
app = Flask(__name__)


# Sentiment Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model is not loaded. Please train the models manually."}), 500

    data = request.json
    review = data.get("review", "")

    if not review:
        return jsonify({"error": "Empty review text provided."}), 400

    review_tfidf = vectorizer.transform([review]).toarray()
    prediction = model.predict(review_tfidf)[0]

    sentiment = "Positive" if prediction == 1 else "Negative" if prediction == 0 else "Neutral"
    return jsonify({"review": review, "sentiment": sentiment})


if __name__ == "__main__":
    app.run(debug=False)
