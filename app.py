from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load models
vectorizer = joblib.load("vectorizer.pkl")
model_nb = joblib.load("model_nb.pkl")
model_pa = joblib.load("model_pa.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    news_text = ""
    selected_model = "nb"   # default model

    if request.method == "POST":
        news_text = request.form["news"]
        selected_model = request.form["model"]

        vectorized_text = vectorizer.transform([news_text])

        if selected_model == "nb":
            model = model_nb
        else:
            model = model_pa

        prediction = model.predict(vectorized_text)[0]

        if hasattr(model, "predict_proba"):
            confidence = round(max(model.predict_proba(vectorized_text)[0]) * 100, 2)
        else:
            confidence = 90

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        news_text=news_text,
        selected_model=selected_model   # ✅ ADD THIS
    )

if __name__ == "__main__":
    app.run(debug=True)