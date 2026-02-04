from flask import Flask, request, jsonify, send_from_directory  
from flask_cors import CORS
import joblib
import pandas as pd
from explain import explain_prediction

app = Flask(__name__)
CORS(app)

# Load model and feature columns
model = joblib.load("efficiency_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


@app.route("/", methods=["GET"])
def serve_html():
    return send_from_directory('static', 'main.html')


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ML service running"})

# Prediction Endpoint
@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Handle GET request (for browser testing)
    if request.method == "GET":
        return jsonify({
            "message": "This endpoint requires POST request with JSON data",
            "example": {
                "temperature": 25,
                "humidity": 60,
                "irradiance": 800
            }
        }), 200
    
    # Handle POST request
    try:
        if not request.json:
            return jsonify({"error": "No JSON data provided"}), 400
        
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        df = df.reindex(columns=feature_columns, fill_value=0)
        
        # Predict efficiency
        efficiency = model.predict(df)[0]
        
        # Simple risk score
        risk_score = round((1 - efficiency) * 100, 2)
        
        # SHAP explanation
        try:
            explanation = explain_prediction(data)
        except Exception as e:
            print(f"Explanation error: {str(e)}")
            explanation = "Explanation unavailable"
        
        # Recommended action
        action = "Immediate maintenance required" if risk_score > 60 else "Monitor closely"
        
        response = {
            "predicted_efficiency": round(float(efficiency), 3),
            "risk_score": risk_score,
            "explanation": explanation,
            "recommended_action": action
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

# Run server
if __name__ == "__main__":
    app.run(debug=True, port=5001)
