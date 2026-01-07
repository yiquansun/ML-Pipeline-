import sys
import pytest
import pandas as pd
import numpy as np
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
from ml.data import process_data
from ml.model import compute_model_metrics, inference

# Setup data for testing
@pytest.fixture
def data():
    # Create a small dummy dataframe that mimics your census data
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
    """ Test the preprocessing function returns expected shapes and types """
    cat_features = ["workclass", "education", "marital-status", "occupation", 
                    "relationship", "race", "sex", "native-country"]
    
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    
    # Check if X is a numpy array (expected type)
    assert isinstance(X, np.ndarray)
    # Check if the number of rows matches the input data
    assert X.shape[0] == len(data)
    # Check if labels were processed
    assert len(y) == len(data)

def test_inference_output(data):
    """ Test that the model inference returns the correct type and size """
    # Load your actual trained model (adjust path if needed)
    model_path = os.path.join("model", "model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # Create a dummy feature row of correct size
        # (Assuming your model expects 108 features after encoding)
        dummy_input = np.zeros((1, 108)) 
        preds = inference(model, dummy_input)
        
        # Check type and shape
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 1
    else:
        pytest.skip("Model file not found, skipping inference test")

def test_compute_metrics():
    """ Test that metrics calculation returns the correct types """
    y = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    
    # Check that metrics are floats
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)