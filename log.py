import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.metrics import roc_auc_score
import joblib

df = pd.read_csv("data.csv")

df = df.drop(columns=['id', 'dataset', 'ca', 'thal', 'slope'])

num_cols = ['trestbps', 'chol', 'thalch', 'oldpeak']
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = ['fbs', 'restecg', 'exang']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df_encoded = pd.get_dummies(df, columns=['sex', 'cp', 'restecg'], drop_first=True)

X = df_encoded.drop("num", axis=1)
Y = df_encoded["num"]
Y = Y.apply(lambda x: 1 if x > 0 else 0)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    penalty='l2',                 # Ridge regularization
    C=0.5,                        # stronger regularization
    class_weight={0: 1.5, 1: 2},  # prioritize CAD
    max_iter=1000
)

model.fit(X_train_scaled, Y_train)

train_pred = model.predict(X_train_scaled)
print("Training Accuracy:", accuracy_score(Y_train, train_pred))
print("Training Recall (CAD):", recall_score(Y_train, train_pred))

test_pred = model.predict(X_test_scaled)
print("Testing Accuracy:", accuracy_score(Y_test, test_pred))
print("Testing Recall (CAD):", recall_score(Y_test, test_pred))

print("Confusion Matrix:\n", confusion_matrix(Y_test, test_pred))
print("Classification Report:\n", classification_report(Y_test, test_pred))

# ---- Threshold tuning ----
y_proba = model.predict_proba(X_test_scaled)[:, 1]

custom_threshold = 0.47
y_pred_thresh = (y_proba >= custom_threshold).astype(int)

print("Confusion Matrix (threshold tuned):\n",
      confusion_matrix(Y_test, y_pred_thresh))
print("Classification Report (threshold tuned):\n",
      classification_report(Y_test, y_pred_thresh))
print("Testing Recall (CAD, threshold tuned):",
      recall_score(Y_test, y_pred_thresh))
roc_auc = roc_auc_score(Y_test, y_proba)
print("ROC-AUC Score:", roc_auc)



joblib.dump(model, "logistic_cad_model.pkl")
joblib.dump(scaler, "logistic_scaler.pkl")
joblib.dump(X.columns, "logistic_feature_columns.pkl")