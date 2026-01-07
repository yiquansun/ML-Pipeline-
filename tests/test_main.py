import sys
sys.path.append(".") # This tells Python to look in the root folder for main.py
from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the Census Income Prediction API!"}

def test_post_predict_lower():
    # Use data known to be <=50K
    data = {
        "age": 20,
        "workclass": "Private",
        "fnlgt": 100000,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 20,
        "native-country": "United-States"
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == "<=50K"

def test_post_predict_higher():
    # Use data known to be >50K by using extreme high-income features
    data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 234721,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 99999,  # This is a very strong indicator for >50K
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    r = client.post("/predict", json=data)
    assert r.status_code == 200
    assert r.json()["prediction"] == ">50K"