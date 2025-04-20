# ✅ File: qnn_model/train_qnn.py
import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===== Function to create a QNN Model =====
def create_qnn_model(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (6, n_qubits)}
    return qml.qnn.TorchLayer(qnode, weight_shapes)

class QNNClassifier(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.qlayer = create_qnn_model(n_qubits)

    def forward(self, x):
        return torch.sigmoid(self.qlayer(x))

# ===== Function to train and save model =====
def train_qnn(X, y, model_path, scaler_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    model = QNNClassifier(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(50):
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = loss_fn(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_pred = model(X_test_tensor).round()
        acc = (test_pred.eq(y_test_tensor)).sum().item() / len(y_test_tensor)
    print(f"✅ Accuracy: {acc:.2f}")

    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

# ====== 1️⃣ DIABETES ======
print("\n🔵 Training QNN model for Diabetes...")
df = pd.read_csv("/Users/shreyabidare/Desktop/diabetes.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
train_qnn(X, y, "qnn_model/qnn_diabetes_model.pkl", "qnn_model/qnn_diabetes_scaler.pkl")

# ====== 2️⃣ HEART DISEASE ======
print("\n❤️ Training QNN model for Heart Disease...")
df1 = pd.read_csv("/Users/shreyabidare/Desktop/heart_cleveland_upload.csv")
X1 = df1.drop(columns=["target"])
y1 = df1["target"]
train_qnn(X1, y1, "qnn_model/qnn_heart_model.pkl", "qnn_model/qnn_heart_scaler.pkl")

# ====== 3️⃣ LUNG CANCER ======
print("\n🫁 Training QNN model for Lung Cancer...")
df2 = pd.read_csv("/Users/shreyabidare/Desktop/survey lung cancer.csv")
df2['LUNG_CANCER'] = df2['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
X2 = df2.drop(columns=["LUNG_CANCER"])
y2 = df2["LUNG_CANCER"]
train_qnn(X2, y2, "qnn_model/qnn_lung_model.pkl", "qnn_model/qnn_lung_scaler.pkl")

print("\n🎉 All 3 QNN models trained and saved successfully!")
