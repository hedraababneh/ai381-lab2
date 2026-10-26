from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "churn_v2.csv"
MODEL_DIR = ROOT / "models"

NUMERIC = ["tenure_months", "monthly_fee", "support_calls"]
CATEGORICAL = ["contract", "internet_service"]
TARGET = "churn"


def build_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def main():
    df = pd.read_csv(DATA)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline().fit(X_train, y_train)
    print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    print(f"Saved {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()