import os
import sys
import pytest
import pandas as pd
import numpy as np
import joblib

# Absolute path resolution for GitHub Actions environment
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ml.data import process_data
from ml.model import compute_model_metrics, inference

@pytest.fixture
def data():
    """ Create a small dummy dataframe for testing """
    df = pd.DataFrame({
        "age": [39, 50],
        "workclass": ["State-gov", "Self-emp-not-inc"],
        "education": ["Bachelors", "Bachelors"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"],
        "salary": ["<=50K", "<=50K"]
    })
    return df

def test_process_data(data):
    """ Test preprocessing function shapes and types """
    cat_features = ["workclass", "education", "marital-status", "occupation", 
                    "relationship", "race", "sex", "native-country"]
    
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    assert isinstance(X, np.ndarray)
    assert X.shape[0] == len(data)
    assert len(y) == len(data)

def test_inference_output(data):
    """ Test model inference returns correct types """
    model_path = os.path.join(root_dir, "model", "model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Assuming your model expects 108 features after encoding
        dummy_input = np.zeros((1, 108)) 
        preds = inference(model, dummy_input)
        
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 1
    else:
        pytest.skip("Model file not found, skipping inference test")

def test_compute_metrics():
    """ Test metrics calculation return types """
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)