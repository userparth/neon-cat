import numpy as np
import pandas as pd
import re
import string
import nltk
import pickle
import os
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

# Categorize Ratings
def categorize_rating(rating):
    if rating >= 4:
        return 1  # Positive
    elif rating == 3:
        return 2  # Neutral
    else:
        return 0  # Negative

# File Paths
DATA_PATH = "data/tripadvisor_hotel_reviews.csv"
MODEL_PATH = "models/svm_sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Load and Process Data
df = pd.read_csv(DATA_PATH)
df['Review'] = df['Review'].apply(clean_text)
df['Rating'] = df['Rating'].apply(categorize_rating)

# Apply TF-IDF transformation
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,1), stop_words='english')
X_tfidf = vectorizer.fit_transform(df['Review'])

# Train Model
model = SVC(probability=True, decision_function_shape='ovr')
model.fit(X_tfidf, df['Rating'])

# Save Model and Vectorizer
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)
with open(VECTORIZER_PATH, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model training completed and saved successfully!")