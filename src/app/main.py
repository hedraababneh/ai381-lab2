from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"
app = FastAPI(title="AI381 Churn API")
model = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_fee: float = Field(gt=0, le=500)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["fiber", "dsl", "none"]
    support_calls: int = Field(ge=0, le=50)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: Customer):
    X = pd.DataFrame([c.model_dump()])
    proba = float(model.predict_proba(X)[0, 1])
    return {"churn": int(proba >= 0.5), "probability": round(proba, 4)}