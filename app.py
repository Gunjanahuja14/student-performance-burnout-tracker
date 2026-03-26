from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route('/')
def home():
    return "Backend is running!"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # If opened in browser (GET request)
    if request.method == 'GET':
        return "Use POST request with JSON data to get prediction"

    try:
        data = request.get_json()

        # Validate input
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

        # Prepare features
        features = [[
            float(data['study_hours']),
            float(data['sleep_hours']),
            float(data['stress_level']),
            float(data['screen_time']),
            float(data['mental_fatigue'])
        ]]

        # Prediction
        prediction = model.predict(features)[0]
        result = "High Burnout" if prediction == 1 else "Low Burnout"

        return jsonify({
            "prediction": result,
            "input": data
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)