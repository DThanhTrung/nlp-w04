from .load_data import (
    load_train_data,
    load_valid_data,
    load_test_data
)

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pandas as pd

def train_embedding_models(
    input_shape: int = 32,
    output_shape: int = 3
) :
    X_train, y_train = load_train_data()
    print(f"Train data.shape: {X_train.shape}")
    print(f"Train label.shape: {y_train.shape}")
    print("Done loading data")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=32, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    print(f"Train accuracy: {accuracy_score(y_train, y_train_pred)}")

    X_val, y_val = load_valid_data()
    X_val_scaled = scaler.transform(X_val)

    y_val_pred = model.predict(X_val_scaled)
    print(f"Val accuracy: {accuracy_score(y_val, y_val_pred)}")

    X_test, test_id = load_test_data()
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    print(f"Test prediction: {y_pred}")

    submission_df = pd.DataFrame({
        "id": test_id,
        "label": y_pred
    }).reset_index(drop=True)
    submission_df.to_csv("w12/submission.csv", index=None)

    print("Submission saved into w12/submission.csv")