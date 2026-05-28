import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Any

from forecast_food_demand import (
    download_dataset,
    load_sales_data,
    build_time_series_features,
)
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
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


ROOT = pathlib.Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "arima_data.xls"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline() -> Dict[str, Any]:
    download_dataset(DATA_PATH)
    df = load_sales_data(DATA_PATH)
    df_features = build_time_series_features(df)

    train_df = df_features.iloc[:-7].copy()
    test_df = df_features.iloc[-7:].copy()
    feature_cols = [c for c in df_features.columns if c not in ["date", "sales"]]

    # regression
    xgb_pred = train_xgboost(train_df, test_df, feature_cols)
    linear_pred = train_linear_regression(train_df, test_df, feature_cols)
    rf_pred = train_random_forest_regressor(train_df, test_df, feature_cols)
    prophet_pred = train_prophet(train_df, periods=len(test_df))

    def eval_arr(a, b):
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        mae = mean_absolute_error(a, b)
        rmse = float(np.sqrt(mean_squared_error(a, b)))
        return mae, rmse

    xgb_mae, xgb_rmse = eval_arr(test_df["sales"].to_numpy(), xgb_pred)
    linear_mae, linear_rmse = eval_arr(test_df["sales"].to_numpy(), linear_pred)
    rf_mae, rf_rmse = eval_arr(test_df["sales"].to_numpy(), rf_pred)
    prophet_mae, prophet_rmse = eval_arr(test_df["sales"].to_numpy(), prophet_pred)

    # classification (small holdout)
    classification_df = df_features.copy()
    classification_threshold = classification_df["sales"].median()
    classification_df["high_demand"] = (
        classification_df["sales"] > classification_threshold
    ).astype(int)
    from sklearn.model_selection import train_test_split

    cls_train, cls_test = train_test_split(
        classification_df,
        test_size=0.25,
        random_state=42,
        stratify=classification_df["high_demand"],
    )

    logistic_pred = train_logistic_regression(cls_train, cls_test, feature_cols, classification_threshold)
    rf_class_pred = train_random_forest_classifier(cls_train, cls_test, feature_cols, classification_threshold)

    cls_test_target = cls_test["high_demand"].to_numpy()
    logistic_metrics = evaluate_classification(cls_test_target, logistic_pred)
    rf_class_metrics = evaluate_classification(cls_test_target, rf_class_pred)

    # Full-dataset cross-validated predictions for unbiased overall confusion matrix
    classification_features = classification_df[feature_cols]
    classification_labels = classification_df["high_demand"].to_numpy()

    # logistic CV predict with standard scaling
    log_pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=200))
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    try:
        log_cv_pred = cross_val_predict(log_pipe, classification_features, classification_labels, cv=5)
    except Exception:
        # fallback to simple predict if cross_val fails on small data
        log_pipe.fit(classification_features, classification_labels)
        log_cv_pred = log_pipe.predict(classification_features)

    try:
        rf_cv_pred = cross_val_predict(rf_clf, classification_features, classification_labels, cv=5)
    except Exception:
        rf_clf.fit(classification_features, classification_labels)
        rf_cv_pred = rf_clf.predict(classification_features)

    full_log_metrics = evaluate_classification(classification_labels, log_cv_pred)
    full_rf_metrics = evaluate_classification(classification_labels, rf_cv_pred)


    metrics = {
        "regression": {
            "XGBoost": {"mae": xgb_mae, "rmse": xgb_rmse},
            "Linear": {"mae": linear_mae, "rmse": linear_rmse},
            "RandomForest": {"mae": rf_mae, "rmse": rf_rmse},
            "Prophet": {"mae": prophet_mae, "rmse": prophet_rmse},
        },
        "classification_threshold": float(classification_threshold),
        "classification": {
            "Logistic": logistic_metrics,
            "RandomForest": rf_class_metrics,
        },
        "full_classification": {
            "Logistic_full_cv": full_log_metrics,
            "RandomForest_full_cv": full_rf_metrics,
        },
    }

    # sanitize metrics for JSON/templating (convert numpy arrays/types to native Python types)
    def sanitize(o):
        if isinstance(o, dict):
            return {k: sanitize(v) for k, v in o.items()}
        if isinstance(o, list):
            return [sanitize(x) for x in o]
        # numpy types
        try:
            import numpy as _np

            if isinstance(o, _np.ndarray):
                return o.tolist()
            if isinstance(o, (_np.int_, _np.int32, _np.int64)):
                return int(o)
            if isinstance(o, (_np.floating, _np.float32, _np.float64)):
                return float(o)
        except Exception:
            pass
        return o

    metrics_clean = sanitize(metrics)

    return {
        "metrics": metrics_clean,
        "forecast_plot": (OUTPUT_DIR / "forecast_results.png").as_posix(),
        "metrics_image": (OUTPUT_DIR / "metrics_summary.png").as_posix(),
    }
