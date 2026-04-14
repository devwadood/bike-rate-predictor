"""FastAPI application for predicting bike resale prices."""

from datetime import datetime
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Advanced Bike Price Predictor API")

model = joblib.load('bike_linear_model.pkl')
model_columns = joblib.load('model_columns.pkl')

class BikeData(BaseModel):  # pylint: disable=too-few-public-methods
    """Request body for a bike price prediction."""

    brand: str
    model_year: int
    engine_capacity: int  # (e.g., 70, 100, 125)
    mileage: int
    condition_score: int
    target_year: Optional[int] = None  # Optional parameter for future prediction

@app.get("/")
def home():
    """Return a simple API health message."""

    return {"message": "Advanced Bike API is Running!"}

@app.post("/predict")
def predict_price(data: BikeData):
    """Predict the bike resale price for the current or requested year."""

    current_year = datetime.now().year
    prediction_year = data.target_year if data.target_year else current_year

    bike_age = prediction_year - data.model_year

    if bike_age < 0:
        raise HTTPException(status_code=400, detail="Target year cannot be older than Model year!")

    input_data = {
        "brand": data.brand,
        "engine_capacity": data.engine_capacity,
        "bike_age": bike_age,
        "mileage": data.mileage,
        "condition_score": data.condition_score
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=['brand'])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)
    predicted_price = round(float(prediction[0]))
    predicted_price = max(predicted_price, 0)

    return {
        "status": "success",
        "predicted_price_rs": predicted_price,
        "calculated_for_year": prediction_year,
        "calculated_bike_age": bike_age,
        "bike_details": data.dict()
    }
