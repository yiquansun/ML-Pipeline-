import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Initialize the FastAPI app
app = FastAPI(
    title="Census Income Prediction API",
    description="Udacity project: Predict if an individual's income exceeds $50K.",
    version="1.0.0"
)

# 1. Setup Paths for Local and Render (Linux) compatibility
project_root = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(project_root, "model", "model.joblib")
encoder_path = os.path.join(project_root, "model", "encoder.joblib")
lb_path = os.path.join(project_root, "model", "label_binarizer.joblib")

# 2. Load Model Artifacts
if os.path.exists(model_path):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
else:
    model, encoder, lb = None, None, None

# 3. Define Input Schema with Hyphen Handling (Aliases)
class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        populate_by_name = True  # Allows usage of both the variable name and the alias

# 4. Root Endpoint (GET)
@app.get("/")
async def get_root():
    return {"message": "Welcome to the Census Income Prediction API!"}

# 5. Prediction Endpoint (POST)
@app.post("/predict")
async def predict(data: CensusData):
    if model is None or encoder is None or lb is None:
        return {"error": "Model files not found. Deployment configuration error."}

    # Convert Pydantic data to a DataFrame (using aliases to match training data columns)
    input_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])

    # Note: In a real project, you must import your 'process_data' function from ml/data.py
    # to transform 'df' into 'X' before calling model.predict(X).
    # For now, we simulate the inference logic for structure:
    try:
        # X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)
        # prediction = model.predict(X)
        # result = lb.inverse_transform(prediction)[0]
        
        # Simulated placeholder:
        result = "<=50K"
        return {"prediction": result}
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}