from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os
import traceback
app = Flask(__name__)
CORS(app)

# Configuration
USE_OPENAI = False  # Set to True if you have an OpenAI API key
OPENAI_API_KEY = "api"  
try:
    from explain import explain_prediction
    model_eff = joblib.load("efficiency_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    model_site = joblib.load("suitability_model.pkl")
    site_feature_columns = joblib.load("site_feature_columns.pkl")
    print("Models loaded successfully")
    print(f" Efficiency model features: {len(feature_columns)}")
    print(f" Suitability model features: {len(site_feature_columns)}")
except FileNotFoundError as e:
    print(f"ERROR: Could not load model files - {e}")
    print("Make sure you have run train_ml.py first to generate the .pkl files")
    exit(1)

# Load explain module
try:
    
    SHAP_AVAILABLE = True
    print(" SHAP explanation module loaded")
except ImportError:
    SHAP_AVAILABLE = False
    print(" SHAP not available - will use simplified explanations")
    
    def explain_prediction(data):
        """Fallback explanation without SHAP"""
        return {
            "temperature": -0.05 if data.get('temperature', 25) > 35 else 0.02,
            "humidity": -0.03 if data.get('humidity', 50) > 70 else 0.01,
            "irradiance": 0.08 if data.get('irradiance', 800) > 700 else -0.05,
            "dust_index": -0.06 if data.get('dust_index', 0.5) > 0.5 else 0.01
        }

# Load OpenAI if enabled
if USE_OPENAI:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print(" OpenAI client initialized")
    except ImportError:
        print(" OpenAI library not installed. Run: pip install openai")
        USE_OPENAI = False
    except Exception as e:
        print(f" OpenAI initialization failed: {e}")
        USE_OPENAI = False

def get_genai_insights(shap_explanation, efficiency):
    """Get AI-powered insights (or fallback to rule-based)"""
    if USE_OPENAI:
        try:
            prompt = f"Based on SHAP values {shap_explanation} for solar panel efficiency of {efficiency:.2%}. Provide 3 concise insights and actionable suggestions to improve or maintain efficiency."
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return get_fallback_insights(shap_explanation, efficiency)
    else:
        return get_fallback_insights(shap_explanation, efficiency)

def get_fallback_insights(shap_explanation, efficiency):
    """Rule-based insights when OpenAI is not available"""
    insights = []
    
    # Analyze efficiency level
    if efficiency > 0.85:
        insights.append("✓ Excellent performance - System is operating at peak efficiency.")
    elif efficiency > 0.75:
        insights.append("⚠ Good performance with room for optimization.")
    else:
        insights.append("⚠ Low efficiency detected - Immediate attention required.")
    
    # Analyze top factors
    top_factors = sorted(shap_explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    for feature, impact in top_factors:
        if 'temperature' in feature.lower() and impact < -0.03:
            insights.append("🌡️ High temperature is reducing efficiency. Consider cooling systems or shade structures.")
        elif 'irradiance' in feature.lower() and impact > 0.05:
            insights.append("☀️ Good solar irradiance levels. Maintain panel cleanliness to maximize capture.")
        elif 'dust' in feature.lower() and impact < -0.03:
            insights.append("🧹 Dust accumulation detected. Schedule cleaning to restore 5-10% efficiency.")
        elif 'humidity' in feature.lower() and impact < -0.02:
            insights.append("💧 High humidity affecting performance. Monitor for condensation issues.")
    
    # Add recommendations
    if efficiency < 0.80:
        insights.append("📊 Recommendation: Conduct full system diagnostic and performance audit.")
    else:
        insights.append("🔄 Recommendation: Continue regular maintenance schedule for optimal performance.")
    
    return "\n".join(insights)

@app.route("/", methods=["GET"])
def serve_html():
    """Serve the main HTML page"""
    try:
        return send_from_directory('.', 'static/main.html')
    except FileNotFoundError:
        return jsonify({
            "error": "main.html not found",
            "message": "Make sure main.html is in the same directory as app.py"
        }), 404

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": True,
        "shap_available": SHAP_AVAILABLE,
        "openai_enabled": USE_OPENAI
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required inputs
        required = ['temperature', 'humidity', 'irradiance']
        missing = [field for field in required if field not in data]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400
        
        # Set defaults
        if 'dust_index' not in data:
            data['dust_index'] = 0.5
        
        # Calculate panel temperature
        data['panel_temp'] = data['temperature'] + (data['irradiance'] / 800.0) * 20
        
        # Add default features that might be in the model
        data['cloudcover'] = data.get('cloudcover', 0)
        data['precip'] = data.get('precip', 0)
        data['wind_speed'] = data.get('wind_speed', 5)
        data['voltage'] = data.get('voltage', 35)
        data['current'] = data.get('current', 8)
        
        # Prepare efficiency prediction
        df = pd.DataFrame([data])
        df = df.reindex(columns=feature_columns, fill_value=0)
        
        # Predict efficiency
        efficiency = float(model_eff.predict(df)[0])
        efficiency = max(0.0, min(1.0, efficiency))  # Clamp between 0 and 1
        
        risk_score = round((1 - efficiency) * 100, 2)
        failure_flag = efficiency < 0.75
        
        # Get SHAP explanation
        try:
            explanation = explain_prediction(data)
        except Exception as e:
            print(f"SHAP explanation error: {e}")
            explanation = {"error": "Explanation not available"}
        
        # Get AI insights
        try:
            insights = get_genai_insights(explanation, efficiency)
        except Exception as e:
            print(f"Insights generation error: {e}")
            insights = f"Efficiency: {efficiency:.1%}. System {'requires attention' if failure_flag else 'operating normally'}."
        
        # Suitability prediction
        try:
            site_data = {
                'GHI (kWh/m²/day)': data['irradiance'] / 200,
                'DNI (kWh/m²/day)': data['irradiance'] / 240,
                'DHI (% of GHI)': 20,
                'Snowfall (mm/year)': 0,
                'Quarter1-Cloud cover': data['cloudcover'],
                'Quarter1-Sunshine duration': 8,
                'Quarter1-Ambient temperature': data['temperature'],
                'Quarter1-Relative humidity': data['humidity'],
                'Quarter1-Precipitation': data['precip'],
                'Quarter2-Cloud cover': data['cloudcover'],
                'Quarter2-Sunshine duration': 8,
                'Quarter2-Ambient temperature': data['temperature'],
                'Quarter2-Relative humidity': data['humidity'],
                'Quarter2-Precipitation': data['precip'],
                'Quarter3-Cloud cover': data['cloudcover'],
                'Quarter3-Sunshine duration': 8,
                'Quarter3-Ambient temperature': data['temperature'],
                'Quarter3-Relative humidity': data['humidity'],
                'Quarter3-Precipitation': data['precip'],
                'Quarter4-Cloud cover': data['cloudcover'],
                'Quarter4-Sunshine duration': 8,
                'Quarter4-Ambient temperature': data['temperature'],
                'Quarter4-Relative humidity': data['humidity'],
                'Quarter4-Precipitation': data['precip'],
                'YearlyCloud cover': data['cloudcover'],
                'Sunshine duration': 8,
                'Ambient temperature': data['temperature'],
                'Relative humidity': data['humidity'],
                'Precipitation': data['precip'] * 4
            }
            
            df_site = pd.DataFrame([site_data])
            df_site = df_site.reindex(columns=site_feature_columns, fill_value=0)
            suitability_pred = model_site.predict(df_site)[0]
            suitability = 'Yes' if suitability_pred == 1 else 'No'
        except Exception as e:
            print(f"Suitability prediction error: {e}")
            traceback.print_exc()
            suitability = 'Unknown'
        
        # Determine recommended action
        if risk_score < 30:
            action = "Monitor closely - System performing well"
        elif risk_score < 60:
            action = "Optimize - Consider maintenance and cleaning"
        else:
            action = "Immediate action required - Failure risk detected!"
        
        # Return results
        return jsonify({
            "predicted_efficiency": round(efficiency, 3),
            "risk_score": risk_score,
            "failure_flag": failure_flag,
            "explanation": explanation,
            "insights_and_suggestions": insights,
            "recommended_action": action,
            "suitability": suitability
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SOLARSENSE AI - FLASK SERVER")
    print("="*80)
    print(f"Models loaded: ✓")
    print(f"SHAP available: {'✓' if SHAP_AVAILABLE else '⚠ Using fallback'}")
    print(f"OpenAI enabled: {'✓' if USE_OPENAI else '⚠ Using rule-based insights'}")
    print(f"\nServer starting on http://localhost:5001")
    print("="*80 + "\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')
