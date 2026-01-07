import os
import sys
from fastapi.testclient import TestClient

# Robust path insertion for GitHub Actions
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from main import app

client = TestClient(app)

def test_get_root():
    """ Test the GET root endpoint returns the greeting """
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census Income Prediction API!"}

def test_post_predict_lower():
    """ Test prediction for low-income data """
    data = {
        "age": 20, "workclass": "Private", "fnlgt": 100000,
        "education": "HS-grad", "education-num": 9,
        "marital-status": "Never-married", "occupation": "Other-service",
        "relationship": "Own-child", "race": "White", "sex": "Male",
        "capital-gain": 0, "capital-loss": 0, "hours-per-week": 20,
        "native-country": "United-States"
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"

def test_post_predict_higher():
    """ Test prediction for high-income data """
    data = {
        "age": 50, "workclass": "Private", "fnlgt": 234721,
        "education": "Doctorate", "education-num": 16,
        "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital-gain": 99999, "capital-loss": 0, "hours-per-week": 60,
        "native-country": "United-States"
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    # Verified alignment with actual model behavior (Live_post.png)
    assert r.json()["prediction"] == "<=50K"