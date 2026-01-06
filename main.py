import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data

# Initialize FastAPI
app = FastAPI()

# Pydantic model for input data
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    # Example data for FastAPI docs
    class Config:
        schema_extra = {
            "example": {
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
        }

# Load artifacts
# We use a path that works both locally and on the server
project_root = os.path.dirname(__file__)
model = joblib.load(os.path.join(project_root, "model/model.joblib"))
encoder = joblib.load(os.path.join(project_root, "model/encoder.joblib"))
lb = joblib.load(os.path.join(project_root, "model/lb.joblib"))

# 1. GET route (Welcome message)
@app.get("/")
async def welcome():
    return {"message": "Welcome to the Census Income Prediction API!"}

# 2. POST route (Inference)
@app.post("/predict")
async def predict(data: CensusData):
    # Convert Pydantic model to DataFrame
    # Using by_alias=True ensures the hyphens match the training columns
    input_df = pd.DataFrame([data.dict(by_alias=True)])
    
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    # Process data using our existing ml.data logic
    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, 
        training=False, encoder=encoder, lb=lb
    )
    
    # Run prediction
    prediction = model.predict(X)
    
    # Convert prediction back to label using LabelBinarizer
    result = lb.inverse_transform(prediction)[0]
    
    return {"prediction": result}