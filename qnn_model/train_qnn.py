# train_qnn.py

import os
import pandas as pd
import pickle
import json
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ✅ Setup
os.makedirs("qnn_model", exist_ok=True)

# ✅ Helper to Train and Evaluate
def train_and_save_qnn(X, y, model_path, scaler_path):
    n_features = X.shape[1]

    # Setup dynamic quantum device
    dev = qml.device("default.qubit", wires=n_features)

    # Define QNode
    def qnode(inputs, weights):
        for i in range(n_features):
            qml.RY(inputs[i] + weights[i], wires=i)
        for i in range(n_features - 1):
            qml.CNOT(wires=[i, i+1])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_features)]

    qnode_torch = qml.QNode(qnode, dev, interface="torch")
    weight_shapes = {"weights": (n_features,)}

    # Define QNN Layer
    qnn_layer = qml.qnn.TorchLayer(qnode_torch, weight_shapes)

    # Full model
    class QuantumNeuralNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.qnn = qnn_layer
            self.fc = nn.Linear(n_features, 1)

        def forward(self, x):
            x = torch.stack([self.qnn(x_i) for x_i in x])

            x = self.fc(x)
            x = torch.sigmoid(x)
            return x


    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Torch Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Model, Optimizer, Loss
    model = QuantumNeuralNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    # Training
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    preds = model(X_test_tensor).detach().numpy()
    preds = (preds > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)

    # Save Model and Scaler
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    return acc

# ========= 1️⃣ QNN DIABETES ========= #
print("\n🔵 Training QNN model for Diabetes...")

df = pd.read_csv("data/diabetes.csv")
X = df[["Glucose", "BMI", "Age", "DiabetesPedigreeFunction"]]
y = df["Outcome"]

acc_diabetes = train_and_save_qnn(X, y, "qnn_model/qnn_diabetes_model.pth", "qnn_model/qnn_diabetes_scaler.pkl")
print(f"✅ QNN Diabetes Model Accuracy: {acc_diabetes:.2f}")

# ========= 2️⃣ QNN HEART ========= #
print("\n❤️ Training QNN model for Heart Disease...")

df1 = pd.read_csv("data/heart_cleveland_upload.csv")
X1 = df1[["age", "sex", "cp", "thalach", "oldpeak", "exang"]]
y1 = df1["condition"]

acc_heart = train_and_save_qnn(X1, y1, "qnn_model/qnn_heart_model.pth", "qnn_model/qnn_heart_scaler.pkl")
print(f"✅ QNN Heart Disease Model Accuracy: {acc_heart:.2f}")

# ========= 3️⃣ QNN LUNG CANCER ========= #
print("\n🫁 Training QNN model for Lung Cancer...")

df2 = pd.read_csv("data/survey lung cancer.csv")
df2.columns = df2.columns.str.strip().str.upper().str.replace(" ", "_")
df2["GENDER"] = df2["GENDER"].map({"M": 1, "F": 0})
df2["LUNG_CANCER"] = df2["LUNG_CANCER"].map({"YES": 1, "NO": 0})
df2.dropna(inplace=True)

X2 = df2[["GENDER", "AGE", "SMOKING", "COUGHING", "SHORTNESS_OF_BREATH", "CHEST_PAIN"]]
y2 = df2["LUNG_CANCER"]

acc_lung = train_and_save_qnn(X2, y2, "qnn_model/qnn_lung_model.pth", "qnn_model/qnn_lung_scaler.pkl")
print(f"✅ QNN Lung Cancer Model Accuracy: {acc_lung:.2f}")

# ========= 🔚 Final Summary ========= #
print("\n📊 FINAL QNN MODEL ACCURACY SUMMARY:")
print(f"🔵 Diabetes Accuracy:       {acc_diabetes:.2f}")
print(f"❤️ Heart Disease Accuracy: {acc_heart:.2f}")
print(f"🫁 Lung Cancer Accuracy:   {acc_lung:.2f}")

# Save Metrics
qnn_metrics = {
    "diabetes": round(acc_diabetes, 4),
    "heart": round(acc_heart, 4),
    "lung": round(acc_lung, 4)
}

with open("qnn_model/qnn_metrics.json", "w") as f:
    json.dump(qnn_metrics, f)

print("\n📁 Saved QNN accuracy metrics to qnn_model/qnn_metrics.json")
