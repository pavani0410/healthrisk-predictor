from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import torch
import torch.nn as nn
import json
from dotenv import load_dotenv
import os
import pennylane as qml
import google.generativeai as genai

# ========= Load Gemini API Key ==========
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

app = Flask(__name__)

# ========= DEFINE MISSING QuantumClassifier ==========
class QuantumClassifier:
    def __init__(self, model=None):
        self.model = model

    def predict(self, X):
        return np.zeros((X.shape[0],), dtype=int)

# ========= DEFINE QNN MODEL ==========
class QNNModel(nn.Module):
    def __init__(self, input_size):
        super(QNNModel, self).__init__()
        self.input_size = input_size
        self.dev = qml.device("default.qubit", wires=input_size)

        def qnode(inputs, weights):
            for i in range(input_size):
                qml.RY(inputs[i] + weights[i], wires=i)
            for i in range(input_size - 1):
                qml.CNOT(wires=[i, i+1])
            return [qml.expval(qml.PauliZ(i)) for i in range(input_size)]

        self.qnode = qml.QNode(qnode, self.dev, interface="torch")
        weight_shapes = {"weights": (input_size,)}
        self.qnn = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.fc = nn.Linear(input_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.stack([self.qnn(x_i) for x_i in x])
        x = self.fc(x)
        x = self.activation(x)
        return x

# ========= LOAD MODELS ==========
diabetes_dl_model = load_model("dl_model/diabetes_model.h5")
diabetes_ml_model = joblib.load("ml_model/ml_diabetes_model.pkl")
diabetes_scaler = joblib.load("ml_model/ml_diabetes_scaler.pkl")

heart_dl_model = load_model("dl_model/heart_model.h5")
heart_ml_model = joblib.load("ml_model/ml_heart_model.pkl")
heart_scaler = joblib.load("ml_model/ml_heart_scaler.pkl")

lung_dl_model = load_model("dl_model/lung_cancer_dl_model.h5")
lung_ml_model = joblib.load("ml_model/lung_cancer_model.pkl")
lung_scaler = joblib.load("ml_model/lung_cancer_scaler.pkl")

qml_diabetes_model = joblib.load("qml_model/qml_diabetes_model.pkl")
qml_diabetes_scaler = joblib.load("qml_model/qml_diabetes_scaler.pkl")

qml_heart_model = joblib.load("qml_model/qml_heart_model.pkl")
qml_heart_scaler = joblib.load("qml_model/qml_heart_scaler.pkl")

qml_lung_model = joblib.load("qml_model/qml_lung_model.pkl")
qml_lung_scaler = joblib.load("qml_model/qml_lung_scaler.pkl")

qnn_diabetes_model = QNNModel(input_size=4)
qnn_diabetes_model.load_state_dict(torch.load("qnn_model/qnn_diabetes_model.pth", map_location=torch.device('cpu')))
qnn_diabetes_model.eval()
qnn_diabetes_scaler = joblib.load("qnn_model/qnn_diabetes_scaler.pkl")

qnn_heart_model = QNNModel(input_size=6)
qnn_heart_model.load_state_dict(torch.load("qnn_model/qnn_heart_model.pth", map_location=torch.device('cpu')))
qnn_heart_model.eval()
qnn_heart_scaler = joblib.load("qnn_model/qnn_heart_scaler.pkl")

qnn_lung_model = QNNModel(input_size=6)
qnn_lung_model.load_state_dict(torch.load("qnn_model/qnn_lung_model.pth", map_location=torch.device('cpu')))
qnn_lung_model.eval()
qnn_lung_scaler = joblib.load("qnn_model/qnn_lung_scaler.pkl")

# ========= LOAD METRICS ==========
with open("ml_model/metrics.json") as f:
    ml_acc = json.load(f)

with open("dl_model/metrics.json") as f:
    dl_acc = json.load(f)

with open("qml_model/qml_metrics.json") as f:
    qml_acc = json.load(f)

with open("qnn_model/qnn_metrics.json") as f:
    qnn_acc = json.load(f)

# ========= HELPER FUNCTIONS ==========
def predict_qnn(model, input_array):
    with torch.no_grad():
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)
        output = model(input_tensor)
        prediction = torch.round(output).cpu().numpy()
    return bool(prediction[0][0])

def generate_llm_recommendation(disease, input_data, risk_models):
    prompt = f"""
    A patient is predicted to be at risk of {disease} based on: {input_data}.
    Confirmed by: {', '.join(risk_models)}.
    Suggest 2–3 medically valid, practical steps one below the other, the patient can take to reduce their risk.
    keep it short but to the point
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

# ========= ROUTES ==========
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes_form():
    return render_template('diabetes_data.html')

@app.route('/heart')
def heart_form():
    return render_template('heart_data.html')

@app.route('/lung')
def lung_form():
    return render_template('lung_data.html')

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    glucose = float(request.form['glucose'])
    bmi = float(request.form['bmi'])
    age = float(request.form['age'])
    dpf = float(request.form['dpf'])

    input_data = np.array([[glucose, bmi, age, dpf]])

    scaled_ml_dl = diabetes_scaler.transform(input_data)
    dl_pred = diabetes_dl_model.predict(scaled_ml_dl, verbose=0)[0][0] > 0.5
    ml_pred = diabetes_ml_model.predict(scaled_ml_dl)[0] == 1

    scaled_qml = qml_diabetes_scaler.transform(input_data)
    qml_pred = qml_diabetes_model.predict(scaled_qml)[0] == 1

    scaled_qnn = qnn_diabetes_scaler.transform(input_data)
    qnn_pred = predict_qnn(qnn_diabetes_model, scaled_qnn)

    at_risk_models = []
    if dl_pred: at_risk_models.append("Deep Learning")
    if ml_pred: at_risk_models.append("Machine Learning")
    if qml_pred: at_risk_models.append("Quantum ML")
    if qnn_pred: at_risk_models.append("Quantum Neural Network")

    help_text = generate_llm_recommendation("Diabetes", input_data.tolist(), at_risk_models) if at_risk_models else ""

    return render_template('result.html'),
        disease="Diabetes",
        dl=dl_pred,
        ml=ml_pred,
        qml=qml_pred,
        qnn=qnn_pred,
        dl_acc=dl_acc["diabetes"],
        ml_acc=ml_acc["diabetes"],
        qml_acc=qml_acc["diabetes"],
        qnn_acc=qnn_acc["diabetes"],
        help_text=help_text
    )

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    thalach = int(request.form['thalach'])
    oldpeak = float(request.form['oldpeak'])
    exang = int(request.form['exang'])

    input_data = np.array([[age, sex, cp, thalach, oldpeak, exang]])

    scaled_ml_dl = heart_scaler.transform(input_data)
    dl_pred = heart_dl_model.predict(scaled_ml_dl, verbose=0)[0][0] > 0.5
    ml_pred = heart_ml_model.predict(scaled_ml_dl)[0] == 1

    scaled_qml = qml_heart_scaler.transform(input_data)
    qml_pred = qml_heart_model.predict(scaled_qml)[0] == 1

    scaled_qnn = qnn_heart_scaler.transform(input_data)
    qnn_pred = predict_qnn(qnn_heart_model, scaled_qnn)

    at_risk_models = []
    if dl_pred: at_risk_models.append("Deep Learning")
    if ml_pred: at_risk_models.append("Machine Learning")
    if qml_pred: at_risk_models.append("Quantum ML")
    if qnn_pred: at_risk_models.append("Quantum Neural Network")

    help_text = generate_llm_recommendation("Heart Disease", input_data.tolist(), at_risk_models) if at_risk_models else ""

    return render_template('result.html',
        disease="Heart Disease",
        dl=dl_pred,
        ml=ml_pred,
        qml=qml_pred,
        qnn=qnn_pred,
        dl_acc=dl_acc["heart"],
        ml_acc=ml_acc["heart"],
        qml_acc=qml_acc["heart"],
        qnn_acc=qnn_acc["heart"],
        help_text=help_text
    )

@app.route('/predict_lung', methods=['POST'])
def predict_lung():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    smoking = int(request.form['smoking'])
    coughing = int(request.form['coughing'])
    chest_pain = int(request.form['chest_pain'])
    shortness_breath = int(request.form['shortness_breath'])

    input_data = np.array([[gender, age, smoking, coughing, chest_pain, shortness_breath]])

    scaled_ml_dl = lung_scaler.transform(input_data)
    dl_pred = lung_dl_model.predict(scaled_ml_dl, verbose=0)[0][0] > 0.5
    ml_pred = lung_ml_model.predict(scaled_ml_dl)[0] == 1

    scaled_qml = qml_lung_scaler.transform(input_data)
    qml_pred = qml_lung_model.predict(scaled_qml)[0] == 1

    scaled_qnn = qnn_lung_scaler.transform(input_data)
    qnn_pred = predict_qnn(qnn_lung_model, scaled_qnn)

    at_risk_models = []
    if dl_pred: at_risk_models.append("Deep Learning")
    if ml_pred: at_risk_models.append("Machine Learning")
    if qml_pred: at_risk_models.append("Quantum ML")
    if qnn_pred: at_risk_models.append("Quantum Neural Network")

    help_text = generate_llm_recommendation("Lung Cancer", input_data.tolist(), at_risk_models) if at_risk_models else ""

    return render_template('result.html',
        disease="Lung Cancer",
        dl=dl_pred,
        ml=ml_pred,
        qml=qml_pred,
        qnn=qnn_pred,
        dl_acc=dl_acc["lung"],
        ml_acc=ml_acc["lung"],
        qml_acc=qml_acc["lung"],
        qnn_acc=qnn_acc["lung"],
        help_text=help_text
    )



# ========= MAIN ==========
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
