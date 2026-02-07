"""
SHAP-based explanation module for solar panel efficiency predictions
"""

import pandas as pd

# Try to import required libraries
try:
    import shap
    import joblib
    SHAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required libraries - {e}")
    print("Run: pip install shap joblib")
    SHAP_AVAILABLE = False

if SHAP_AVAILABLE:
    try:
        # Load efficiency model & features
        model = joblib.load("efficiency_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        print("✓ SHAP TreeExplainer initialized successfully")
        
    except FileNotFoundError as e:
        print(f"Error: Model files not found - {e}")
        print("Make sure you have run train_ml.py first")
        SHAP_AVAILABLE = False
    except Exception as e:
        print(f"Error initializing SHAP: {e}")
        SHAP_AVAILABLE = False

def explain_prediction(input_data: dict):
    """
    Generate SHAP-based explanation for a prediction
    
    Args:
        input_data: Dictionary with input features
        
    Returns:
        Dictionary mapping feature names to their SHAP values
    """
    if not SHAP_AVAILABLE:
        # Fallback to simple rule-based explanation
        return get_simple_explanation(input_data)
    
    try:
        # Prepare input dataframe
        df = pd.DataFrame([input_data])
        df = df.reindex(columns=feature_columns, fill_value=0)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(df)
        
        # Convert to dictionary
        if isinstance(shap_values, list):
            # For some tree models, shap_values might be a list
            shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
        
        explanation = {
            col: float(shap_values[0][i]) 
            for i, col in enumerate(feature_columns)
        }
        
        return explanation
        
    except Exception as e:
        print(f"SHAP calculation error: {e}")
        return get_simple_explanation(input_data)

def get_simple_explanation(input_data: dict):
    """
    Fallback explanation when SHAP is not available
    Uses simple rules to estimate feature importance
    """
    explanation = {}
    
    # Temperature impact (negative if too high)
    temp = input_data.get('temperature', 25)
    if temp > 35:
        explanation['temperature'] = -0.05 * ((temp - 35) / 10)
    elif temp < 15:
        explanation['temperature'] = -0.02 * ((15 - temp) / 10)
    else:
        explanation['temperature'] = 0.01
    
    # Humidity impact (negative if too high)
    humidity = input_data.get('humidity', 50)
    if humidity > 70:
        explanation['humidity'] = -0.03 * ((humidity - 70) / 30)
    else:
        explanation['humidity'] = 0.005
    
    # Irradiance impact (positive if good)
    irradiance = input_data.get('irradiance', 800)
    if irradiance > 700:
        explanation['irradiance'] = 0.08 * (irradiance / 1000)
    else:
        explanation['irradiance'] = -0.05 * (1 - irradiance / 1000)
    
    # Dust impact (negative)
    dust = input_data.get('dust_index', 0.5)
    explanation['dust_index'] = -0.06 * dust
    
    # Panel temp impact (derived from temperature and irradiance)
    panel_temp = input_data.get('panel_temp', temp + 20)
    if panel_temp > 60:
        explanation['panel_temp'] = -0.04 * ((panel_temp - 60) / 20)
    else:
        explanation['panel_temp'] = 0.01
    
    return explanation

def get_top_features(explanation: dict, n: int = 5):
    """
    Get the top N features by absolute SHAP value
    
    Args:
        explanation: Dictionary of feature SHAP values
        n: Number of top features to return
        
    Returns:
        List of (feature, value) tuples sorted by absolute importance
    """
    sorted_features = sorted(
        explanation.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    return sorted_features[:n]

def format_explanation(explanation: dict, top_n: int = 5):
    """
    Format explanation as human-readable text
    
    Args:
        explanation: Dictionary of feature SHAP values
        top_n: Number of top features to include
        
    Returns:
        Formatted string explanation
    """
    top_features = get_top_features(explanation, top_n)
    
    lines = [f"Top {top_n} factors affecting efficiency:"]
    lines.append("")
    
    for i, (feature, value) in enumerate(top_features, 1):
        impact = "increases" if value > 0 else "decreases"
        lines.append(f"{i}. {feature}: {impact} efficiency by {abs(value):.4f}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    # Test the explanation function
    test_data = {
        'temperature': 35,
        'humidity': 65,
        'irradiance': 850,
        'dust_index': 0.3
    }
    
    print("Testing SHAP explanation module...")
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print("\nTest input:")
    for k, v in test_data.items():
        print(f"  {k}: {v}")
    
    print("\nExplanation:")
    explanation = explain_prediction(test_data)
    for feature, value in sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:8]:
        print(f"  {feature:20s}: {value:+.4f}")
    
    print("\nFormatted explanation:")
    print(format_explanation(explanation))
