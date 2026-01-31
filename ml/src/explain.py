import shap
import joblib
import pandas as pd

# Load model & feature order
model = joblib.load("efficiency_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

explainer = shap.TreeExplainer(model)

def explain_prediction(input_data: dict):
    """
    Returns SHAP values for given input dictionary
    """
    df = pd.DataFrame([input_data])
    # Reorder columns, fill missing with 0
    df = df.reindex(columns=feature_columns, fill_value=0)

    shap_values = explainer.shap_values(df)

    # Convert to simple dict: {feature: impact}
    explanation = {col: float(shap_values[0][i]) for i, col in enumerate(feature_columns)}
    return explanation
