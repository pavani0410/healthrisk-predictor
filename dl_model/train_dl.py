# train_dl.py - Deep Learning Model Training Script using selected features

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow import keras
import joblib
import json

# Create output directory
os.makedirs("dl_model", exist_ok=True)

# ==================== Diabetes Model ====================
print("\n🩸 Training DL model for Diabetes...")

df_diabetes = pd.read_csv("data/diabetes.csv")
X_d = df_diabetes[["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]]
y_d = df_diabetes["Outcome"]

scaler_d = StandardScaler()
X_d_scaled = scaler_d.fit_transform(X_d)

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d_scaled, y_d, test_size=0.2, stratify=y_d, random_state=42
)

model_d = Sequential([
    Input(shape=(X_train_d.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_d.fit(X_train_d, y_train_d, epochs=100, verbose=0)

loss_d, acc_d = model_d.evaluate(X_test_d, y_test_d, verbose=0)
print(f"✅ Diabetes DL Model Accuracy: {acc_d:.2f}")

model_d.save("dl_model/diabetes_model.h5")
joblib.dump(scaler_d, "dl_model/diabetes_scaler.pkl")

# ==================== Heart Disease Model ====================
print("\n❤️ Training DL model for Heart Disease...")

df_heart = pd.read_csv("data/heart_cleveland_upload.csv")
X_h = df_heart[["age", "sex", "cp", "thalach", "oldpeak", "exang"]]
y_h = df_heart["condition"]

scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h_scaled, y_h, test_size=0.2, stratify=y_h, random_state=42
)

model_h = Sequential([
    Input(shape=(X_train_h.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_h.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_h.fit(X_train_h, y_train_h, epochs=100, verbose=0)

loss_h, acc_h = model_h.evaluate(X_test_h, y_test_h, verbose=0)
print(f"✅ Heart Disease DL Model Accuracy: {acc_h:.2f}")

model_h.save("dl_model/heart_model.h5")
joblib.dump(scaler_h, "dl_model/heart_scaler.pkl")

# ==================== Lung Cancer Model ====================
print("\n🫁 Training DL model for Lung Cancer...")

df_l = pd.read_csv("data/survey lung cancer.csv")
df_l.columns = df_l.columns.str.strip().str.upper().str.replace(" ", "_")

df_l["GENDER"] = df_l["GENDER"].map({"M": 1, "F": 0})
df_l["LUNG_CANCER"] = df_l["LUNG_CANCER"].map({"YES": 1, "NO": 0})
df_l.dropna(inplace=True)

X_l = df_l[["GENDER", "AGE", "SMOKING", "COUGHING", "SHORTNESS_OF_BREATH", "CHEST_PAIN"]]
y_l = df_l["LUNG_CANCER"]

scaler_l = StandardScaler()
X_l_scaled = scaler_l.fit_transform(X_l)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_l_scaled, y_l, test_size=0.2, random_state=42
)

model_l = Sequential([
    Input(shape=(X_train_l.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_l.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_l.fit(X_train_l, y_train_l, epochs=100, verbose=0)

loss_l, acc_l = model_l.evaluate(X_test_l, y_test_l, verbose=0)
print(f"✅ Lung Cancer DL Model Accuracy: {acc_l:.2f}")

model_l.save("dl_model/lung_cancer_dl_model.h5")
joblib.dump(scaler_l, "dl_model/lung_cancer_scaler.pkl")

# ==================== Save Accuracy Metrics ====================
dl_metrics = {
    "diabetes": round(acc_d, 4),
    "heart": round(acc_h, 4),
    "lung": round(acc_l, 4)
}

with open("dl_model/metrics.json", "w") as f:
    json.dump(dl_metrics, f)

print("\n📁 Saved DL accuracy metrics to dl_model/metrics.json")
