from fastapi.testclient import TestClient
from main import app


def test_sanity_root():
    """Check if root returns 200."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


def test_sanity_predict_schema():
    """Check if predict endpoint schema works."""
    with TestClient(app) as client:
        sample = {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
        response = client.post("/predict", json=sample)
        assert response.status_code == 200
        assert "prediction" in response.json()


if __name__ == "__main__":
    print("Running sanity checks...")
    test_sanity_root()
    test_sanity_predict_schema()
    print("Sanity checks passed!")
