import joblib
import numpy as np
import pandas as pd
import re
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}})  # Allow only the frontend origin

# Load the trained model & scaler
model = joblib.load("phishing_model_best.pkl")
scaler = joblib.load("scaler.pkl")  # If you used feature scaling

# Feature extraction function
def extract_features(url):
    parsed_url = urlparse(url)
    
    features = {
        "url_length": len(url),
        "num_subdomains": len(parsed_url.netloc.split(".")) - 1,
        "num_special_chars": len(re.findall(r"[@_!#$%^&*()<>?/\|}{~:]", url)),
        "is_https": 1 if parsed_url.scheme == "https" else 0,
        "has_phishing_keyword": int(any(keyword in url.lower() for keyword in ["login", "banking", "secure", "verify", "account", "update", "paypal"]))
    }
    
    return np.array(list(features.values())).reshape(1, -1)

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("url")
        if not url:
            return render_template("index.html", url=None, result="No URL provided")
        
        features = extract_features(url)
        features_scaled = scaler.transform(features)  
        prediction = model.predict(features_scaled)[0]

        result = "Phishing" if prediction == 1 else "Legitimate"
        return render_template("index.html", url=url, result=result)

    return render_template("index.html", url=None, result=None)

# API Endpoint for external requests
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    
    features = extract_features(url)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    result = "Phishing" if prediction == 1 else "Legitimate"
    return jsonify({"url": url, "prediction": result})

if __name__ == "__main__":
    app.run(debug=True, port=5000)