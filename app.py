from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

# Home page (HTML UI)
@app.route('/')
def home():
    return render_template("index.html")


# JSON API (for frontend / Postman / teammate)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        required_fields = [
            "study_hours",
            "sleep_hours",
            "stress_level",
            "screen_time",
            "mental_fatigue"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"})

        features = [[
            float(data['study_hours']),
            float(data['sleep_hours']),
            float(data['stress_level']),
            float(data['screen_time']),
            float(data['mental_fatigue'])
        ]]

        prediction = model.predict(features)[0]
        result = "High Burnout" if prediction == 1 else "Low Burnout"

        return jsonify({
            "prediction": result,
            "input": data
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# FORM ROUTE (for browser UI)
@app.route('/predict_form', methods=['POST'])
def predict_form():
    try:
        features = [[
            float(request.form['study_hours']),
            float(request.form['sleep_hours']),
            float(request.form['stress_level']),
            float(request.form['screen_time']),
            float(request.form['mental_fatigue'])
        ]]

        prediction = model.predict(features)[0]
        result = "High Burnout" if prediction == 1 else "Low Burnout"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)