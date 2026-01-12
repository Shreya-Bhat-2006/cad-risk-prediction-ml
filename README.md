# CAD Risk Prediction using Machine Learning

## Overview

This project focuses on building a **machine learning–based decision support system** to predict the **early risk of Coronary Artery Disease (CAD)** using **non-invasive clinical and lifestyle data**.

The system is designed **to assist doctors**, not replace them, by providing an **early risk signal** and a **clear explanation of contributing factors**, helping prioritize further medical evaluation.

---

## Where This Model Fits in a Hospital

This model sits **between basic patient screening and advanced diagnostic tests**.

### Typical Clinical Flow (Simplified)

1. Patient arrives with symptoms (or during routine checkup)
2. Doctor collects basic, non-invasive data:
    - Age
    - Blood pressure
    - Cholesterol
    - Blood sugar
    - ECG summary
    - Exercise-induced symptoms
3. **ML model runs at this stage**
4. Model outputs:
    - CAD risk level
    - Key contributing factors
5. Doctor decides:
    - Monitor / lifestyle advice
    - Further tests (ECG, CCTA, angiography)

👉 The model **does not skip medical tests**

👉 It **helps prioritize patients earlier**

---

## Problem Statement

**Build a machine learning model that predicts the early risk of Coronary Artery Disease (CAD) using non-invasive clinical and lifestyle data to support doctors in decision-making.**

---

## Project Goal

The goal of this project is to:

- Predict whether a patient is at **low risk or high risk** of CAD
- Provide a **brief, interpretable explanation** for the prediction
- Assist doctors in **early risk assessment**, not diagnosis

### Input

- Non-invasive patient data (clinical + lifestyle features)

### Output

- **Binary CAD risk prediction**
    - `0` → Low risk
    - `1` → High risk
- **Key factors contributing to risk**

---

## Project Scope

- Uses **tabular data** only
- Focuses on **early risk prediction**
- Uses **non-invasive clinical and lifestyle features**
- Applies **traditional machine learning models**
- Emphasizes **interpretability**

---

## Out of Scope

- No medical images (CT, MRI, angiography)
- No deep learning models
- No treatment or medication recommendations
- No replacement of clinical judgment

---

## Dataset

This project uses the **UCI Heart Disease dataset**, which contains patient-level clinical attributes commonly used in CAD research.

📎 Dataset link:

[https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?resource=download](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?resource=download)

---

## Models Implemented

Two different models were trained and evaluated independently:

### Logistic Regression (Interpretable Baseline)

- Regularized (L2 / Ridge)
- Class weighting to prioritize CAD cases
- Threshold tuning for higher recall
- ROC-AUC evaluation

### Random Forest (Performance-Oriented)

- Tuned depth and sample parameters
- Class weighting to reduce false negatives
- Threshold tuning to improve CAD recall

Both models are saved separately and can be used independently for prediction.

---

## Model Performance (Key Metrics Only)

### 🔹 Logistic Regression (Improved Version)

- **Test Accuracy:** ~81%
- **Test Recall (CAD):** ~90%
- **ROC-AUC:** ~0.91
- Strong interpretability
- Suitable as a clinical baseline model

### 🔹 Random Forest (Improved Version)

- **Test Accuracy:** ~84%
- **Test Recall (CAD):** ~92–94%
- Better performance on complex patterns
- Slightly less interpretable than Logistic Regression

👉 **Recall is prioritized** because missing a high-risk CAD patient is more dangerous than a false alarm.

---

## Explainability

- Logistic Regression coefficients are used to identify **top contributing features**
- Helps answer:
    
    > “Why was this patient predicted as high risk?”
    > 
- Supports trust and transparency in clinical use

---

## How to Use

- Models are saved using `joblib`
- Separate prediction scripts are provided:
    - `logistic_predict_cad.py`
    - `rf_predict_cad.py`
- New patient data can be passed to generate:
    - Risk probability
    - Risk classification
    - Feature contribution summary

---

## Future Scope

- Incorporate ECG signal data
- Explore deep learning with medical imaging
- Integrate SHAP for model-agnostic explanations
- Build a simple web or desktop interface for clinicians

---

## ⚠️ Disclaimer

This project is **for educational and research purposes only**.

It is **not a medical diagnostic tool** and must not be used as a replacement for professional medical advice.
