from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    vectorized = vectorizer.transform([news])
    prediction = model.predict(vectorized)[0]

    result = "Real News" if prediction == 1 else "Fake News"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)