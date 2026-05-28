# Food & Restaurant Demand Forecasting

This project uses a live public restaurant sales dataset from GitHub.
The dataset is downloaded automatically when the forecasting script runs.

## Dataset
- Source: `https://github.com/Ultraopxt/ARIMA-time-series-analysis-forecasting-restaurant-sales`
- File: `arima_data.xls`
- Columns: `date`, `sales`

## Run
1. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Run the forecast pipeline:
   ```bash
   python forecast_food_demand.py
   ```

## Output
## Output
- `data/arima_data.xls` - downloaded dataset
- `output/forecast_results.png` - actual vs predicted demand plot
- `output/metrics_summary.png` - model performance summary (image)
- Terminal output includes MAE and RMSE for all regression models and classification metrics

## Project layout
- `models/` - modular model implementations (`regressors.py`, `classifiers.py`)
- `output/` - generated plots and metrics images
- `data/` - raw dataset
# Food-Demand-Forecasting
AI-based restaurant demand forecasting and inventory optimization project
