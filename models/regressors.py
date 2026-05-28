import xgboost as xgb
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import List
import pandas as pd
import numpy as np


def train_xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(train_df[features], train_df["sales"])
    return model.predict(test_df[features])


def train_linear_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    model = LinearRegression()
    model.fit(train_df[features], train_df["sales"])
    return model.predict(test_df[features])


def train_random_forest_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(train_df[features], train_df["sales"])
    return model.predict(test_df[features])


def train_prophet(train_df: pd.DataFrame, periods: int) -> np.ndarray:
    prophet_df = train_df[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return forecast["yhat"].iloc[-periods:].to_numpy()
