import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib
import json

# Ensure output directory exists
os.makedirs("ml_model", exist_ok=True)

# ========= 1️⃣ DIABETES ========= #
print("\n🔵 Training ML model for Diabetes...")

df = pd.read_csv("data/diabetes.csv")
X = df[["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler_diabetes = StandardScaler()
X_train_scaled = scaler_diabetes.fit_transform(X_train)
X_test_scaled = scaler_diabetes.transform(X_test)

model_diabetes = RandomForestClassifier(n_estimators=100, random_state=42)
model_diabetes.fit(X_train_scaled, y_train)

y_pred = model_diabetes.predict(X_test_scaled)
acc_diabetes = accuracy_score(y_test, y_pred)
print(f"✅ Diabetes ML Model Accuracy: {acc_diabetes:.2f}")

with open("ml_model/ml_diabetes_model.pkl", "wb") as f:
    pickle.dump(model_diabetes, f)
with open("ml_model/ml_diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler_diabetes, f)

# ========= 2️⃣ HEART DISEASE ========= #
print("\n❤️ Training ML model for Heart Disease...")

df1 = pd.read_csv("data/heart_cleveland_upload.csv")
X1 = df1[["age", "sex", "cp", "thalach", "oldpeak", "exang"]]
y1 = df1["condition"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, stratify=y1, test_size=0.2, random_state=42)

scaler_heart = StandardScaler()
X1_train_scaled = scaler_heart.fit_transform(X1_train)
X1_test_scaled = scaler_heart.transform(X1_test)

model_heart = RandomForestClassifier(random_state=42)
model_heart.fit(X1_train_scaled, y1_train)

y1_pred = model_heart.predict(X1_test_scaled)
acc_heart = accuracy_score(y1_test, y1_pred)
print(f"✅ Heart Disease ML Model Accuracy: {acc_heart:.2f}")

with open("ml_model/ml_heart_model.pkl", "wb") as f:
    pickle.dump(model_heart, f)
with open("ml_model/ml_heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler_heart, f)

# ========= 3️⃣ LUNG CANCER ========= #
print("\n🫁 Training ML model for Lung Cancer...")

df2 = pd.read_csv("data/survey lung cancer.csv")
df2.columns = df2.columns.str.strip().str.upper().str.replace(" ", "_")

df2["GENDER"] = df2["GENDER"].map({"M": 1, "F": 0})
df2["LUNG_CANCER"] = df2["LUNG_CANCER"].map({"YES": 1, "NO": 0})
df2.dropna(inplace=True)

X2 = df2[["GENDER", "AGE", "SMOKING", "COUGHING", "SHORTNESS_OF_BREATH", "CHEST_PAIN"]]
y2 = df2["LUNG_CANCER"]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

scaler_lung = StandardScaler()
X2_train_scaled = scaler_lung.fit_transform(X2_train)
X2_test_scaled = scaler_lung.transform(X2_test)

model_lung = LogisticRegression(max_iter=1000)
model_lung.fit(X2_train_scaled, y2_train)

acc_lung = model_lung.score(X2_test_scaled, y2_test)
print(f"✅ Lung Cancer ML Model Accuracy: {acc_lung:.2f}")

joblib.dump(model_lung, "ml_model/lung_cancer_model.pkl")
joblib.dump(scaler_lung, "ml_model/lung_cancer_scaler.pkl")

# ========= 🔚 Final Summary ========= #
print("\n📊 FINAL MODEL ACCURACY SUMMARY:")
print(f"🔵 Diabetes Accuracy:       {acc_diabetes:.2f}")
print(f"❤️ Heart Disease Accuracy: {acc_heart:.2f}")
print(f"🫁 Lung Cancer Accuracy:   {acc_lung:.2f}")

# ========= ✅ Save ML Accuracies to metrics.json ========= #
ml_metrics = {
    "diabetes": round(acc_diabetes, 4),
    "heart": round(acc_heart, 4),
    "lung": round(acc_lung, 4)
}

with open("ml_model/metrics.json", "w") as f:
    json.dump(ml_metrics, f)

print("\n📁 Saved ML accuracy metrics to ml_model/metrics.json")