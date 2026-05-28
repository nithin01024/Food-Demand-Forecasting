import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd
from typing import List


def train_logistic_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], threshold: float) -> np.ndarray:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    y_train = (train_df["sales"] > threshold).astype(int)
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def train_random_forest_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], threshold: float) -> np.ndarray:
    y_train = (train_df["sales"] > threshold).astype(int)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_df[features], y_train)
    return model.predict(test_df[features])


def evaluate_classification(actual: np.ndarray, predicted: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=[0, 1]),
        "classification_report": classification_report(actual, predicted, labels=[0, 1], zero_division=0),
    }
    return metrics
