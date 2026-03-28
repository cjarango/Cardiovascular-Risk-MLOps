from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("app/model.joblib")
app = FastAPI()

class Input(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Input):
    X = np.array(data.features).reshape(1, -1)
    proba = model.predict_proba(X)[0][1]
    return {"heart_disease_probability": proba, "prediction": int(proba > 0.5)}