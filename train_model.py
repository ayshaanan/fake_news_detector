import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier

# Load dataset
df = pd.read_csv("fake_or_real_news.csv")

X = df["text"]
y = df["label"]

# Convert labels to binary
y = y.map({"FAKE": 0, "REAL": 1})

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Passive Aggressive
pa_model = PassiveAggressiveClassifier(max_iter=50)
pa_model.fit(X_train, y_train)

# Save
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(nb_model, "model_nb.pkl")
joblib.dump(pa_model, "model_pa.pkl")

print("Models trained and saved successfully.")