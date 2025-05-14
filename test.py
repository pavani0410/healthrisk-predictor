import unittest
from app import app

class FlaskAppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Personalized Health Risk Predictor', response.data)

    def test_diabetes_prediction(self):
        response = self.app.post('/diabetes', data={
            'pregnancies': 1,
            'glucose': 120,
            'bp': 70,
            'skin': 20,
            'insulin': 85,
            'bmi': 24.5,
            'dpf': 0.5,
            'age': 30
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Diabetes', response.data)

    def test_heart_prediction(self):
        response = self.app.post('/heart', data={
            'age': 45,
            'sex': 1,
            'cp': 3,
            'trestbps': 130,
            'chol': 250,
            'fbs': 0,
            'restecg': 1,
            'thalach': 180,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 2,
            'ca': 0,
            'thal': 2
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Heart Disease', response.data)

    def test_lung_prediction(self):
        response = self.app.post('/lung', data={
            'age': 50,
            'smoking': 1,
            'yellow_fingers': 1,
            'anxiety': 0,
            'peer_pressure': 1,
            'chronic_disease': 0,
            'fatigue': 1,
            'allergy': 0,
            'wheezing': 1,
            'alcohol': 0,
            'coughing': 1,
            'shortness_of_breath': 1,
            'swallowing_difficulty': 0,
            'chest_pain': 1
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Lung Cancer', response.data)

if __name__ == '__main__':
    unittest.main()
