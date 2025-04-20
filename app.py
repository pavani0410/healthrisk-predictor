from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# ========== LOAD MODELS & SCALERS ==========

# Diabetes
diabetes_dl_model = load_model("dl_model/diabetes_model.h5")
diabetes_ml_model = joblib.load("ml_model/ml_diabetes_model.pkl")
diabetes_scaler = joblib.load("ml_model/ml_diabetes_scaler.pkl")

# Heart
heart_dl_model = load_model("dl_model/heart_model.h5")
heart_ml_model = joblib.load("ml_model/ml_heart_model.pkl")
heart_scaler = joblib.load("ml_model/ml_heart_scaler.pkl")

# Lung
lung_dl_model = load_model("dl_model/lung_cancer_dl_model.h5")
lung_ml_model = joblib.load("ml_model/lung_cancer_model.pkl")
lung_scaler = joblib.load("ml_model/lung_cancer_scaler.pkl")

# Load accuracies
with open("ml_model/metrics.json") as f:
    ml_acc = json.load(f)

with open("dl_model/metrics.json") as f:
    dl_acc = json.load(f)

# ========== ROUTES ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET'])
def diabetes_form():
    return render_template('diabetes_data.html')

@app.route('/heart', methods=['GET'])
def heart_form():
    return render_template('heart_data.html')

@app.route('/lung', methods=['GET'])
def lung_form():
    return render_template('lung_data.html')

# ========== PREDICTION ROUTES ==========

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    glucose = float(request.form['glucose'])
    bmi = float(request.form['bmi'])
    age = float(request.form['age'])
    dpf = float(request.form['dpf'])

    input_data = np.array([[glucose, bmi, age, dpf]])
    scaled_data = diabetes_scaler.transform(input_data)

    dl_pred = diabetes_dl_model.predict(scaled_data)[0][0] > 0.5
    ml_pred = diabetes_ml_model.predict(scaled_data)[0] == 1

    help_text = "Maintain a healthy diet, exercise regularly, and consult a doctor for proper medication." if dl_pred or ml_pred else ""

    return render_template('result.html', disease="Diabetes", dl=dl_pred, ml=ml_pred,
                           dl_acc=dl_acc["diabetes"], ml_acc=ml_acc["diabetes"], help_text=help_text)

@app.route('/predict_heart', methods=['POST'])
def predict_heart():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    thalach = int(request.form['thalach'])
    oldpeak = float(request.form['oldpeak'])
    exang = int(request.form['exang'])

    input_data = np.array([[age, sex, cp, thalach, oldpeak, exang]])
    scaled_data = heart_scaler.transform(input_data)

    dl_pred = heart_dl_model.predict(scaled_data)[0][0] > 0.5
    ml_pred = heart_ml_model.predict(scaled_data)[0] == 1

    help_text = "Reduce stress, control blood pressure, eat a heart-healthy diet, and see a cardiologist." if dl_pred or ml_pred else ""

    return render_template('result.html', disease="Heart Disease", dl=dl_pred, ml=ml_pred,
                           dl_acc=dl_acc["heart"], ml_acc=ml_acc["heart"], help_text=help_text)

@app.route('/lung_predict', methods=['POST'])
def predict_lung():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    smoking = int(request.form['smoking'])
    coughing = int(request.form['coughing'])
    chest_pain = int(request.form['chest_pain'])
    shortness_breath = int(request.form['shortness_breath'])

    input_data = np.array([[gender, age, smoking, coughing, shortness_breath, chest_pain]])
    scaled_data = lung_scaler.transform(input_data)

    dl_pred = lung_dl_model.predict(scaled_data)[0][0] > 0.5
    ml_pred = lung_ml_model.predict(scaled_data)[0] == 1

    help_text = "Avoid smoking, reduce exposure to pollutants, and get regular screenings from a specialist." if dl_pred or ml_pred else ""

    return render_template('result.html', disease="Lung Cancer", dl=dl_pred, ml=ml_pred,
                           dl_acc=dl_acc["lung"], ml_acc=ml_acc["lung"], help_text=help_text)

# ========== MAIN ==========

if __name__ == '__main__':
    app.run(debug=True)
