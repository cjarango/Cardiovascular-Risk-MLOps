from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd



model = joblib.load("app/model.joblib")



app = FastAPI(title="API de Riesgo Cardiovascular")



class Input(BaseModel):
    features: list



COLUMNAS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"
]


@app.post("/predict")
def predict(data: Input):
    
    df = pd.DataFrame([data.features], columns=COLUMNAS)

    # Predecimos usando el DataFrame
    proba = model.predict_proba(df)[0][1]

    return {
        "heart_disease_probability": float(proba),
        "prediction": int(proba > 0.5)
    }