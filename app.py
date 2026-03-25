from flask import Flask, request, render_template
import joblib
import os
import math

app = Flask(__name__)

base_dir = os.path.dirname(__file__)

vectorizer = joblib.load(os.path.join(base_dir, "vectorizer.pkl"))
nb_model = joblib.load(os.path.join(base_dir, "model_nb.pkl"))
pa_model = joblib.load(os.path.join(base_dir, "model_pa.pkl"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    news_text = ""
    selected_model = "nb"
    confidence = ""

    if request.method == "POST":
        news_text = request.form.get("news", "")
        selected_model = request.form.get("model", "nb")

        if news_text.strip():
            vect = vectorizer.transform([news_text])
            
            if selected_model == "nb":
                model = nb_model
                pred = model.predict(vect)[0]
                proba = model.predict_proba(vect)[0]
                confidence = round(max(proba) * 100, 2)
            else:
                model = pa_model
                pred = model.predict(vect)[0]
                decision = model.decision_function(vect)[0]
                prob = 1 / (1 + math.exp(-decision)) if decision > -100 else 0
                confidence = round(max(prob, 1 - prob) * 100, 2)

            prediction = int(pred)

    return render_template("index.html", 
                           prediction=prediction, 
                           news_text=news_text, 
                           selected_model=selected_model, 
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)