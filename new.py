import numpy as np
import pandas as pd
import re
import string
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords


# Run the script
if __name__ == '__main__':
    # Download stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


    # Text Cleaning Function
    def clean_text(text):
        text = text.lower()
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = " ".join(word for word in text.split() if word not in stop_words)
        return text


    # Load Dataset
    file_path = "fastreviews.csv"
    data = pd.read_csv(file_path)
    data.dropna(subset=['Review Text'], inplace=True)
    data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0)
    data['Cleaned_Review'] = data['Review Text'].apply(clean_text)

    # Check if numerical features are useful
    print(data[['Delivery Time (min)', 'Customer Service Rating', 'Sentiment']].corr())

    # Train-Test Split (with stratification)
    X = data[['Cleaned_Review', 'Delivery Time (min)', 'Customer Service Rating']]
    y = data['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TF-IDF Vectorization with bigrams
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_text_vec = vectorizer.fit_transform(X_train['Cleaned_Review']).toarray()
    X_test_text_vec = vectorizer.transform(X_test['Cleaned_Review']).toarray()

    # Standardize Numerical Features
    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train[['Delivery Time (min)', 'Customer Service Rating']])
    X_test_num_scaled = scaler.transform(X_test[['Delivery Time (min)', 'Customer Service Rating']])

    # Combine Text + Numerical Features
    X_train_final = np.hstack((X_train_text_vec, X_train_num_scaled))
    X_test_final = np.hstack((X_test_text_vec, X_test_num_scaled))


    # Model Training Function
    def train_and_evaluate(model, model_name):
        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n{model_name} Accuracy: {accuracy:.3f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))


    # Train Different Models with Fixes
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', C=0.1, max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_split=10, min_samples_leaf=5,
                                                class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                        random_state=42),
        "SVM": SVC(kernel='linear', C=1, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, reg_lambda=1, eval_metric='logloss',
                                 use_label_encoder=False)
    }

    # Run Models
    for name, model in models.items():
        train_and_evaluate(model, name)

    pipeline = make_pipeline(TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
                             LogisticRegression(class_weight='balanced', C=0.1, max_iter=500))
    cv_scores = cross_val_score(pipeline, data['Cleaned_Review'], data['Sentiment'], cv=5, scoring='accuracy')

    print("\nCross-validation mean accuracy:", round(cv_scores.mean(), 3))

