from flask import Flask, request, jsonify
import numpy as np
import joblib


model_data = joblib.load("svm_model.pkl")
weights, bias, scaler = model_data["weights"], model_data["bias"], model_data["scaler"]

app = Flask(__name__)

def predict_digit(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    output = np.sign(np.dot(features, weights) - bias)
    return int(output) if output == 1 else 0

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  
    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400
    
    prediction = predict_digit(data["features"])
    return jsonify({"prediction": prediction})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running! Use /predict with a POST request."})

if __name__ == "__main__":
    app.run(debug=True)