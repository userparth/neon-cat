import numpy as np
import pandas as pd
import re
import string
import nltk
import pickle
import os
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Download stopwords
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))


# Text Cleaning Function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


def categorize_rating(rating):
    if rating >= 4:
        return 1  # Positive
    elif rating == 3:
        return 2  # Neutral
    else:
        return 0  # Negative


# Initialize Flask app
app = Flask(__name__)

# Model Paths
MODEL_PATH = "svm_sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
TRAIN_NEW_MODEL = 1


def load_or_train_model():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and TRAIN_NEW_MODEL == 0:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        with open(VECTORIZER_PATH, "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        print("Loaded existing model and vectorizer.")
    else:
        print("Training new model...")
        df = pd.read_csv('tripadvisor_hotel_reviews.csv')
        df['Review'] = df['Review'].apply(clean_text)
        df['Rating'] = df['Rating'].apply(categorize_rating)

        # Apply TF-IDF transformation (better generalization)
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
        X_tfidf = vectorizer.fit_transform(df['Review'])

        # Train SVM with probability enabled
        model = SVC(probability=True, decision_function_shape='ovr')
        model.fit(X_tfidf, df['Rating'])

        # Save the model and vectorizer
        with open(MODEL_PATH, "wb") as model_file:
            pickle.dump(model, model_file)
        with open(VECTORIZER_PATH, "wb") as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        print("Model trained and saved.")


load_or_train_model()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        review = data.get("review", "")
        cleaned_review = clean_text(review)
        review_tfidf = vectorizer.transform([cleaned_review]).toarray()

        # Get prediction and probabilities
        confidence_scores = model.predict_proba(review_tfidf)[0]
        confidence = max(confidence_scores)
        prediction = np.argmax(confidence_scores)

        # Generate chatbot response
        if prediction == 1:
            sentiment = "Positive"
            response = "Glad you had a great experience! 😊"
        elif prediction == 0:
            sentiment = "Negative"
            response = "I'm sorry you had a bad experience. 😞"
        else:
            sentiment = "Neutral"
            response = "Thanks for your feedback! Let us know if we can help."

        return jsonify({
            "review": review,
            "sentiment": sentiment,
            "confidence": confidence,
            "response": response
        })

    except Exception as e:
        return jsonify({"error": str(e)})  # Debugging output


if __name__ == "__main__":
    app.run(debug=False)
