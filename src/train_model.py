import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocessing import preprocess_data   # our preprocessor
from utils import save_model
from evaluate import evaluate_model
from xgboost import XGBClassifier

# --- Paths ---
DATA_PATH = "../data/processed/final_loan_data.csv"
MODEL_PATH = "../models/credit_risk_model.pkl"

def load_data():
    """Load the processed dataset."""
    df = pd.read_csv(DATA_PATH)
    return df

def build_pipeline():
    """Build pipeline with preprocessing + model."""
    preprocessor = preprocess_data()
    model = xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline

def main():
    # Load data
    df = load_data()

    # Split features/target
    X = df.drop(columns=["default payment next month"])
    y = df["default payment next month"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline
    pipeline = build_pipeline()

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_proba)
    print("Model Performance:", metrics)

    # Save model
    save_model(pipeline, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()



