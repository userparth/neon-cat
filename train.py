import numpy as np
import pandas as pd
import re
import string
import nltk
import pickle
import os
import gdown
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Download stopwords
try:
    stop_words = set(nltk.corpus.stopwords.words('english'))
except LookupError:
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
MODEL_PATH = "models/sgd_classifier_sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Google Drive File ID
FILE_ID = "1FGjqzu67Wvsv4Xdz9pbWTu6WXpQ0wD3q"  # Replace with your actual File ID
OUTPUT_PATH = "data/tripadvisor_hotel_reviews.csv"

# Download from Google Drive
if not os.path.exists(OUTPUT_PATH):
    print("Downloading dataset from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT_PATH, quiet=False)
else:
    print("Dataset already exists, skipping download.")

# Load and Process Data
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("Cleaning text...")
tqdm.pandas()
df['Review'] = df['Review'].apply(clean_text)
df['Rating'] = df['Rating'].apply(categorize_rating)

# Apply TF-IDF transformation
print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    sublinear_tf=True,  # Log scaling for better term frequency representation
    max_df=0.9,  # Ignore terms that appear in more than 90% of docs
    min_df=5,  # Ignore terms that appear in fewer than 5 docs
    max_features=5000,  # Limit vocab size for efficiency
    ngram_range=(1, 2),
    stop_words='english'  # Remove common stopwords
)
X_tfidf = vectorizer.fit_transform(tqdm(df['Review'], desc="Vectorizing", unit="rows"))

# Train Model
print("Training model...")
model = SGDClassifier(loss='hinge', max_iter=1, tol=None, random_state=42)  # Train in steps

# Train with live progress bar
num_epochs = 10  # You can increase epochs if needed
for epoch in tqdm(range(1, num_epochs + 1), desc="Training Progress", unit="epoch"):
    model.partial_fit(X_tfidf, df['Rating'], classes=np.array([0, 1, 2]))  # Partial fit updates live!

# Save Model and Vectorizer
os.makedirs("models", exist_ok=True)
with open(MODEL_PATH, "wb") as model_file:
    pickle.dump(model, model_file)
with open(VECTORIZER_PATH, "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model training completed and saved successfully!")
