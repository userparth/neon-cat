# Sentiment Analysis API - Machine Learning Project

## 📌 Overview
Welcome to my **first machine learning project**! 🚀 I built a **Sentiment Analysis API** that predicts whether a review is **Positive, Neutral, or Negative** based on text input. This is not a chatbot—just an **ML-powered API** that analyzes text sentiment.

This project helped me understand **text preprocessing, model training, and deployment** using Python and Flask. If you're new to ML, follow along to see what I learned! 😃

---

## 🎯 Features
✔️ **Predicts sentiment** (Positive, Neutral, Negative) for given text.  
✔️ **REST API built with Flask** to process text requests.  
✔️ **Machine Learning Model (SGDClassifier with TF-IDF)** trained on real hotel reviews.  
✔️ **Optimized for efficiency and step-wise training updates.**  
✔️ **Pre-trained model is saved using Pickle**, so no need to retrain on every run.  

---

## 🛠 Tech Stack
- **Python** - Primary language for ML & API development.  
- **Flask** - Lightweight API framework.  
- **Scikit-Learn** - ML model training.  
- **NLTK** - Text preprocessing (stopwords, punctuation removal).  
- **TF-IDF Vectorizer** - Converts text into numerical vectors.  
- **SGDClassifier (Stochastic Gradient Descent)** - Faster, incremental training model.  
- **Pickle** - Saves and loads trained models efficiently.  

---

## 📂 Project Structure
```
├── data
│   ├── tripadvisor_hotel_reviews.csv  # Dataset used for training
├── models
│   ├── sgd_classifier_sentiment_model.pkl  # Trained ML model
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
├── api.py                             # Flask API script
├── train.py                           # Model training script
├── README.md                          # Project documentation
```

---

## 📊 Model Performance
| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|---------|-----------|--------|----------|
| **SGDClassifier**   | **88.9%** | **88.0%** | **88.5%** | **88.2%** |
| **SVM (SVC)**       | 89.2%    | 88.5%     | 89.1%  | 88.8%    |
| **Random Forest**   | 86.5%    | 85.8%     | 86.0%  | 85.9%    |
| **Logistic Regression** | 83.2% | 82.5%     | 83.0%  | 82.7%    |

✅ **SGDClassifier was chosen** due to its fast, incremental training capabilities while maintaining high accuracy!

---

## 🚀 How to Run the Project
### 1️⃣ Install Dependencies
```bash
pip install flask scikit-learn pandas nltk pickle-mixin tqdm gdown
```

### 2️⃣ Train the Model (if not already trained)
```bash
python train.py
```
This will train an **SGDClassifier** model and save it as `sgd_classifier_sentiment_model.pkl`.

### 3️⃣ Start the API Server
```bash
python api.py
```
The API will start running at: **`http://127.0.0.1:5000/predict`**

### 4️⃣ Test the API with Sample Data
Use **Postman**, **cURL**, or Python requests to test:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"review": "The hotel was fantastic! Loved the stay."}'
```
✅ Expected Response:
```json
{"review": "The hotel was fantastic! Loved the stay.", "sentiment": "Positive"}
```

---

Would love to hear your thoughts! Drop a comment or suggestion! 🚀
