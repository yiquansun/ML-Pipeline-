import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import inference


app = FastAPI(
    title="Census Income Prediction API",
    description="API for predicting census income categories.",
    version="1.0.0"
)


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
        populate_by_name = True
        json_schema_extra = {
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


model = None
encoder = None
lb = None


def load_artifacts():
    """Helper to load artifacts using relative paths."""
    global model, encoder, lb
    if model is None:
        model = joblib.load("model/model.joblib")
        encoder = joblib.load("model/encoder.joblib")
        lb = joblib.load("model/lb.joblib")


@app.on_event("startup")
async def startup_event():
    load_artifacts()


@app.get("/")
async def get_root():
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    # Ensure artifacts are loaded
    load_artifacts()

    input_df = pd.DataFrame([data.dict(by_alias=True)])
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction_raw = inference(model, X)
    prediction_label = lb.inverse_transform(prediction_raw)[0]
    return {"prediction": prediction_label}
