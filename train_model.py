import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# -----------------------
# LOAD DATA
# -----------------------
data = pd.read_csv("fake_or_real_news.csv")

# Combine title + text
data["content"] = data["title"] + " " + data["text"]

# -----------------------
# CLEAN TEXT
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data["content"] = data["content"].apply(clean_text)

# -----------------------
# SPLIT DATA
# -----------------------
X = data["content"]
y = data["label"]

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=5000,
    ngram_range=(1,2)
)

X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# -----------------------
# TRAIN MODEL
# -----------------------
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# -----------------------
# EVALUATE
# -----------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100)

# -----------------------
# SAVE MODEL
# -----------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and Vectorizer saved successfully!")