import numpy as np
import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords


def train_models():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Load dataset
    file_path = "fastreviews.csv"
    data = pd.read_csv(file_path)

    # Drop missing values
    data.dropna(subset=['Review Text'], inplace=True)

    # Convert Rating to Binary Sentiment
    data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0)
    print(data['Sentiment'])

    # 🔹 Text Cleaning Function
    def clean_text(text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = " ".join(word for word in text.split() if word not in stop_words)
        return text

    # Apply text cleaning
    data['Cleaned_Review'] = data['Review Text'].apply(clean_text)
    print(data['Cleaned_Review'])


    # ✅ 1. Split BEFORE TF-IDF
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data['Cleaned_Review'], data['Sentiment'], test_size=0.2, random_state=42
    )

    # ✅ 2. Apply TF-IDF with Bigrams & Increased Features
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,3))  # Unigrams + Bigrams
    X_train_text_vec = vectorizer.fit_transform(X_train_text).toarray()
    X_test_text_vec = vectorizer.transform(X_test_text).toarray()

    # ✅ 3. Use More Features (Include Rating)
    selected_features = ['Rating', 'Delivery Time (min)', 'Customer Service Rating']
    X_numerical = data[selected_features]

    # Split numerical features correctly
    X_train_num, X_test_num = train_test_split(X_numerical, test_size=0.2, random_state=42)

    # Normalize Numerical Features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # ✅ 4. Combine Text + Numerical Features
    X_train = np.hstack((X_train_text_vec, X_train_num_scaled))
    X_test = np.hstack((X_test_text_vec, X_test_num_scaled))

    print("Train-Test Overlap Check:", set(X_train_text) & set(X_test_text))

    # ✅ 5. Train & Evaluate Models
    models = {
        "Logistic Regression": LogisticRegression(C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1),
        "SVM": SVC(kernel='rbf', C=1.0),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, scale_pos_weight=1.5, eval_metric='logloss')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{name} Accuracy: {accuracy:.3f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

# # Run the script
# if __name__ == '__main__':
#     train_models()
