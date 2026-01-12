import pandas as pd
import joblib
import numpy as np

# Load saved logistic regression files
model = joblib.load("logistic_cad_model.pkl")
scaler = joblib.load("logistic_scaler.pkl")
feature_columns = joblib.load("logistic_feature_columns.pkl")

def predict_cad_logistic(new_patient):
    df = pd.DataFrame([new_patient])

    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=['sex', 'cp', 'restecg'], drop_first=True)

    # Align columns
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df_encoded)
  
    # Predict probability
    prob = model.predict_proba(df_scaled)[0][1]

    # Risk category
    if prob >= 0.6:
        risk = "HIGH RISK"
    elif prob >= 0.3:
        risk = "MODERATE RISK"
    else:
        risk = "LOW RISK"

    # Explanation
    coef_df = pd.DataFrame({
        "Feature": feature_columns,
        "Impact": model.coef_[0] * df_encoded.iloc[0]
    })

    top_factors = coef_df.reindex(
        coef_df["Impact"].abs().sort_values(ascending=False).index
    ).head(5)

    print("\n=== Logistic Regression CAD Prediction ===")
    print(f"CAD Probability: {prob:.2f}")
    print(f"Risk Level: {risk}")
    print("\nTop contributing factors:")
    for _, row in top_factors.iterrows():
        if row["Impact"] != 0:
            print("-", row["Feature"])

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

predict_cad_logistic(new_patient)
