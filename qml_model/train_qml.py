# train_qml.py

import os
import pandas as pd
import pickle
import json
import pennylane as qml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# ✅ Setup
os.makedirs("qml_model", exist_ok=True)

# ✅ Quantum Device
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def quantum_circuit(inputs):
    qml.AngleEmbedding(inputs, wires=[0, 1, 2], rotation='Y')
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RY(0.5, wires=0)
    qml.RY(0.5, wires=1)
    qml.RY(0.5, wires=2)
    return [qml.expval(qml.PauliZ(i)) for i in range(3)]

# quantum device will be initialized dynamically inside the model class
class QuantumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_features):
        self.n_features = n_features
        self.weights = np.random.randn(n_features)
        self.dev = qml.device("default.qubit", wires=n_features)

    def quantum_circuit(self, inputs):
        @qml.qnode(self.dev)
        def circuit(x):
            qml.AngleEmbedding(x, wires=range(self.n_features), rotation='Y')
            for i in range(self.n_features - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_features)]
        return circuit(inputs)

    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = []
        for x in X:
            qc_out = self.quantum_circuit(x * self.weights)
            pred = np.sign(np.sum(qc_out))
            preds.append(1 if pred >= 0 else 0)
        return np.array(preds)


# ========= 1️⃣ QML DIABETES ========= #
print("\n🔵 Training QML model for Diabetes...")

df = pd.read_csv("data/diabetes.csv")
X = df[["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

scaler_diabetes = StandardScaler()
X_train_scaled = scaler_diabetes.fit_transform(X_train)
X_test_scaled = scaler_diabetes.transform(X_test)

model_diabetes = QuantumClassifier(n_features=X_train_scaled.shape[1])

model_diabetes.fit(X_train_scaled, y_train)

y_pred = model_diabetes.predict(X_test_scaled)
acc_diabetes = accuracy_score(y_test, y_pred)
print(f"✅ Diabetes QML Model Accuracy: {acc_diabetes:.2f}")

with open("qml_model/qml_diabetes_model.pkl", "wb") as f:
    pickle.dump(model_diabetes, f)
with open("qml_model/qml_diabetes_scaler.pkl", "wb") as f:
    pickle.dump(scaler_diabetes, f)

# ========= 2️⃣ QML HEART ========= #
print("\n❤️ Training QML model for Heart Disease...")

df1 = pd.read_csv("data/heart_cleveland_upload.csv")
X1 = df1[["age", "sex", "cp", "thalach", "oldpeak", "exang"]]
y1 = df1["condition"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, stratify=y1, test_size=0.2, random_state=42)

scaler_heart = StandardScaler()
X1_train_scaled = scaler_heart.fit_transform(X1_train)
X1_test_scaled = scaler_heart.transform(X1_test)

model_heart = QuantumClassifier(n_features=X1_train_scaled.shape[1])
model_heart.fit(X1_train_scaled, y1_train)


y1_pred = model_heart.predict(X1_test_scaled)
acc_heart = accuracy_score(y1_test, y1_pred)
print(f"✅ Heart Disease QML Model Accuracy: {acc_heart:.2f}")

with open("qml_model/qml_heart_model.pkl", "wb") as f:
    pickle.dump(model_heart, f)
with open("qml_model/qml_heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler_heart, f)

# ========= 3️⃣ QML LUNG CANCER ========= #
print("\n🫁 Training QML model for Lung Cancer...")

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

model_lung = QuantumClassifier(n_features=X2_train_scaled.shape[1])
model_lung.fit(X2_train_scaled, y2_train)


y2_pred = model_lung.predict(X2_test_scaled)
acc_lung = accuracy_score(y2_test, y2_pred)
print(f"✅ Lung Cancer QML Model Accuracy: {acc_lung:.2f}")

with open("qml_model/qml_lung_model.pkl", "wb") as f:
    pickle.dump(model_lung, f)
with open("qml_model/qml_lung_scaler.pkl", "wb") as f:
    pickle.dump(scaler_lung, f)

# ========= 🔚 Final Summary ========= #
print("\n📊 FINAL QML MODEL ACCURACY SUMMARY:")
print(f"🔵 Diabetes Accuracy:       {acc_diabetes:.2f}")
print(f"❤️ Heart Disease Accuracy: {acc_heart:.2f}")
print(f"🫁 Lung Cancer Accuracy:   {acc_lung:.2f}")

qml_metrics = {
    "diabetes": round(acc_diabetes, 4),
    "heart": round(acc_heart, 4),
    "lung": round(acc_lung, 4)
}

with open("qml_model/qml_metrics.json", "w") as f:
    json.dump(qml_metrics, f)

print("\n📁 Saved QML accuracy metrics to qml_model/qml_metrics.json")
