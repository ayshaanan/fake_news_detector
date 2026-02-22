from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)

# -----------------------
# USER MODEL
# -----------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    password = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -----------------------
# LOAD DATASET
# -----------------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data[["text", "label"]]

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -----------------------
# ROUTES
# -----------------------

@app.route("/")
def home():
    return render_template("index.html", accuracy=round(accuracy * 100, 2))

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    vectorized = vectorizer.transform([news])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized).max()

    result = "Real News" if prediction == 1 else "Fake News"
    confidence = round(probability * 100, 2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        accuracy=round(accuracy * 100, 2),
    )

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and user.password == request.form["password"]:
            login_user(user)
            return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

# -----------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)