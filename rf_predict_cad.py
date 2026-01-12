import pandas as pd
import joblib

# Load saved Random Forest files
model = joblib.load("rf_cad_model.pkl")
feature_columns = joblib.load("rf_feature_columns.pkl")

def predict_cad_rf(new_patient):
    df = pd.DataFrame([new_patient])

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=['sex', 'cp', 'restecg'], drop_first=True)

    # Align columns
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict probability
    prob = model.predict_proba(df_encoded)[0][1]

    # Risk category
    if prob >= 0.6:
        risk = "HIGH RISK"
        advice = "Immediate cardiac evaluation recommended"
    elif prob >= 0.3:
        risk = "MODERATE RISK"
        advice = "Close monitoring advised"
    else:
        risk = "LOW RISK"
        advice = "Lifestyle management suggested"

    print("\n=== Random Forest CAD Prediction ===")
    print(f"CAD Probability: {prob:.2f}")
    print(f"Risk Level: {risk}")
    print("Recommendation:", advice)

# Example patient
new_patient = {
    "age": 55,
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": 145,
    "chol": 240,
    "fbs": False,
    "restecg": "normal",
    "thalch": 150,
    "exang": True,
    "oldpeak": 2.1
}

predict_cad_rf(new_patient)
