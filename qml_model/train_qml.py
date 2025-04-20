import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
from tensorflow.keras import layers  # Import layers from tensorflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Import LabelEncoder
import pandas as pd
import joblib


# 📌 Load all datasets
df_diabetes = pd.read_csv("/Users/shreyabidare/Desktop/diabetes.csv")
df_heart = pd.read_csv("/Users/shreyabidare/Desktop/heart_cleveland_upload.csv")
df_lung = pd.read_csv("/Users/shreyabidare/Desktop/survey lung cancer.csv")

# ---------------------------------
# 🧠 QNN Setup
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

def circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (6, n_qubits, 3)}
qnode = qml.QNode(circuit, dev, interface="tf")
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits)

# 🔄 Utility to build QML model
def build_qml_model(input_dim):
    model = tf.keras.models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(n_qubits),
        qlayer,
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------------
# 💉 1. Diabetes Dataset
X_d = df_diabetes.drop(columns=["Outcome"])
y_d = df_diabetes["Outcome"]

scaler_d = StandardScaler()
X_d_scaled = scaler_d.fit_transform(X_d)
joblib.dump(scaler_d, "qml_model/diabetes_scaler.pkl")

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d_scaled, y_d, test_size=0.2, random_state=42)

model_d = build_qml_model(X_train_d.shape[1])
print("🩸 Training QML model for Diabetes...")
model_d.fit(X_train_d, y_train_d, epochs=10, verbose=1)
model_d.save("qml_model/diabetes_qml_model.h5")

# ---------------------------------
# ❤️ 2. Heart Disease Dataset
X_h = df_heart.drop(columns=["condition"])
y_h = df_heart["condition"]

scaler_h = StandardScaler()
X_h_scaled = scaler_h.fit_transform(X_h)
joblib.dump(scaler_h, "qml_model/heart_scaler.pkl")

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h_scaled, y_h, test_size=0.2, random_state=42)

model_h = build_qml_model(X_train_h.shape[1])
print("❤️ Training QML model for Heart Disease...")
model_h.fit(X_train_h, y_train_h, epochs=10, verbose=1)
model_h.save("qml_model/heart_qml_model.h5")

# ---------------------------------
# 🫁 3. Lung Cancer Dataset
df_lung["GENDER"] = LabelEncoder().fit_transform(df_lung["GENDER"])
df_lung["LUNG_CANCER"] = df_lung["LUNG_CANCER"].map({"NO": 0, "YES": 1})

X_l = df_lung.drop(columns=["LUNG_CANCER"])
y_l = df_lung["LUNG_CANCER"]

scaler_l = StandardScaler()
X_l_scaled = scaler_l.fit_transform(X_l)
joblib.dump(scaler_l, "qml_model/lung_cancer_scaler.pkl")

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l_scaled, y_l, test_size=0.2, random_state=42)

model_l = build_qml_model(X_train_l.shape[1])
print("🫁 Training QML model for Lung Cancer...")
model_l.fit(X_train_l, y_train_l, epochs=10, verbose=1)
model_l.save("qml_model/lung_qml_model.h5")
