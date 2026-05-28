import os
import pathlib
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/Ultraopxt/"
    "ARIMA-time-series-analysis-forecasting-restaurant-sales/master/arima_data.xls"
)
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
DATA_PATH = DATA_DIR / "arima_data.xls"
PLOT_PATH = pathlib.Path(__file__).resolve().parent / "forecast_results.png"


def download_dataset(path: pathlib.Path) -> None:
    if path.exists():
        print(f"Dataset already exists at {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from live source: {RAW_DATA_URL}")
    urllib.request.urlretrieve(RAW_DATA_URL, str(path))
    print(f"Saved dataset to {path}")


def load_sales_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "sales"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["dayofweek"] = df2["date"].dt.dayofweek
    df2["day"] = df2["date"].dt.day
    df2["month"] = df2["date"].dt.month
    df2["quarter"] = df2["date"].dt.quarter
    df2["year"] = df2["date"].dt.year
    df2["is_weekend"] = (df2["dayofweek"] >= 5).astype(int)
    df2["weekofyear"] = df2["date"].dt.isocalendar().week.astype(int)

    for lag in [1, 2, 3, 7]:
        df2[f"lag_{lag}"] = df2["sales"].shift(lag)

    for window in [3, 7]:
        df2[f"roll_mean_{window}"] = df2["sales"].shift(1).rolling(window).mean()
        df2[f"roll_std_{window}"] = df2["sales"].shift(1).rolling(window).std()

    df2 = df2.dropna().reset_index(drop=True)
    return df2


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse
from models.regressors import (
    train_xgboost,
    train_linear_regression,
    train_random_forest_regressor,
    train_prophet,
)
from models.classifiers import (
    train_logistic_regression,
    train_random_forest_classifier,
    evaluate_classification,
)


def plot_forecasts(
    df: pd.DataFrame,
    xgb_pred: np.ndarray,
    prophet_pred: np.ndarray,
    linear_pred: np.ndarray,
    rf_pred: np.ndarray,
) -> None:
    test = df.iloc[-len(xgb_pred) :].copy()
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["sales"], marker="o", label="Actual Sales")
    plt.plot(test["date"], xgb_pred, marker="x", label="XGBoost Forecast")
    plt.plot(test["date"], linear_pred, marker="^", label="Linear Regression")
    plt.plot(test["date"], rf_pred, marker="s", label="RandomForest Forecast")
    plt.plot(test["date"], prophet_pred, marker="D", label="Prophet Forecast")
    plt.title("Restaurant Demand Forecast vs Actual Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales Volume")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Saved forecast plot to {PLOT_PATH}")


def save_metrics_image(metrics: dict, output_path: pathlib.Path) -> None:
    lines = []
    lines.append("Regression metrics:\n")
    for name, vals in metrics.get("regression", {}).items():
        lines.append(f"{name}: MAE={vals['mae']:.2f}, RMSE={vals['rmse']:.2f}")
    lines.append("")
    lines.append(f"Classification threshold: {metrics.get('classification_threshold', 0):.0f}")
    lines.append("")
    for name, vals in metrics.get("classification", {}).items():
        lines.append(f"{name}:")
        lines.append(f"  Accuracy: {vals['accuracy']:.2f}")
        lines.append(f"  Precision: {vals['precision']:.2f}")
        lines.append(f"  Recall: {vals['recall']:.2f}")
        lines.append(f"  F1: {vals['f1']:.2f}")
        lines.append(f"  Confusion matrix: {vals['confusion_matrix']}")
        lines.append("")
    text = "\n".join(lines)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    ax.text(0, 1, text, fontsize=10, va="top", family="monospace")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved metrics summary to {output_path}")


def main() -> None:
    download_dataset(DATA_PATH)
    df = load_sales_data(DATA_PATH)
    print(f"Loaded dataset with {len(df)} rows")

    df_features = build_time_series_features(df)
    print(f"Built features, resulting rows: {len(df_features)}")

    train_df = df_features.iloc[:-7].copy()
    test_df = df_features.iloc[-7:].copy()
    feature_cols = [c for c in df_features.columns if c not in ["date", "sales"]]

    threshold = df_features["sales"].quantile(0.90)
    print(f"Using dataset 90th percentile threshold {threshold:.0f} for binary classification")

    xgb_pred = train_xgboost(train_df, test_df, feature_cols)
    linear_pred = train_linear_regression(train_df, test_df, feature_cols)
    rf_pred = train_random_forest_regressor(train_df, test_df, feature_cols)
    prophet_pred = train_prophet(train_df, periods=len(test_df))

    xgb_mae, xgb_rmse = evaluate_predictions(test_df["sales"].to_numpy(), xgb_pred)
    linear_mae, linear_rmse = evaluate_predictions(test_df["sales"].to_numpy(), linear_pred)
    rf_mae, rf_rmse = evaluate_predictions(test_df["sales"].to_numpy(), rf_pred)
    prophet_mae, prophet_rmse = evaluate_predictions(test_df["sales"].to_numpy(), prophet_pred)

    print("\nRegression model performance:")
    print(f"XGBoost   MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")
    print(f"Linear    MAE: {linear_mae:.2f}, RMSE: {linear_rmse:.2f}")
    print(f"RandomForest MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
    print(f"Prophet   MAE: {prophet_mae:.2f}, RMSE: {prophet_rmse:.2f}")

    classification_df = df_features.copy()
    classification_threshold = classification_df["sales"].median()
    classification_df["high_demand"] = (classification_df["sales"] > classification_threshold).astype(int)
    cls_train, cls_test = train_test_split(
        classification_df,
        test_size=0.25,
        random_state=42,
        stratify=classification_df["high_demand"],
    )
    print(f"\nUsing dataset median threshold {classification_threshold:.0f} for classification labels")

    logistic_pred = train_logistic_regression(cls_train, cls_test, feature_cols, classification_threshold)
    rf_class_pred = train_random_forest_classifier(cls_train, cls_test, feature_cols, classification_threshold)
    cls_test_target = cls_test["high_demand"].to_numpy()

    logistic_metrics = evaluate_classification(cls_test_target, logistic_pred)
    rf_class_metrics = evaluate_classification(cls_test_target, rf_class_pred)

    print("\nClassification performance (high demand label):")
    print("Logistic Regression:")
    print(f"  Accuracy: {logistic_metrics['accuracy']:.2f}")
    print(f"  Precision: {logistic_metrics['precision']:.2f}")
    print(f"  Recall: {logistic_metrics['recall']:.2f}")
    print(f"  F1: {logistic_metrics['f1']:.2f}")
    print("  Confusion matrix:")
    print(logistic_metrics["confusion_matrix"])
    print(logistic_metrics["classification_report"])

    print("RandomForest Classifier:")
    print(f"  Accuracy: {rf_class_metrics['accuracy']:.2f}")
    print(f"  Precision: {rf_class_metrics['precision']:.2f}")
    print(f"  Recall: {rf_class_metrics['recall']:.2f}")
    print(f"  F1: {rf_class_metrics['f1']:.2f}")
    print("  Confusion matrix:")
    print(rf_class_metrics["confusion_matrix"])
    print(rf_class_metrics["classification_report"])

    # save metrics summary image
    metrics = {
        "regression": {
            "XGBoost": {"mae": xgb_mae, "rmse": xgb_rmse},
            "Linear": {"mae": linear_mae, "rmse": linear_rmse},
            "RandomForest": {"mae": rf_mae, "rmse": rf_rmse},
            "Prophet": {"mae": prophet_mae, "rmse": prophet_rmse},
        },
        "classification_threshold": classification_threshold,
        "classification": {
            "Logistic": logistic_metrics,
            "RandomForest": rf_class_metrics,
        },
    }
    save_metrics_image(metrics, pathlib.Path("output") / "metrics_summary.png")

    plot_forecasts(df, xgb_pred, prophet_pred, linear_pred, rf_pred)


if __name__ == "__main__":
    main()
