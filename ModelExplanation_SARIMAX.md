# Explaining the SARIMAX model
Below is the Python code to perform demand forecasting using **SARIMAX** (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors). SARIMAX is a statistical time series forecasting model suitable for datasets with trend and seasonality.

---

### Steps:
1. **Data Preparation:** Filter the data for each SKU and preprocess it.
2. **Train SARIMAX Model for Each SKU:** Fit a SARIMAX model for each SKU using past data.
3. **Forecast for the Next Day:** Predict the next day's demand for each SKU.

---

### Code:

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("sku_data.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Sort the data
data = data.sort_values(by=['seller_sku_code', 'order_date'])

# Initialize an empty list to store forecasts
all_forecasts = []

# Loop through each SKU
sku_codes = data['seller_sku_code'].unique()

for sku in sku_codes:
    # Filter data for the current SKU
    sku_data = data[data['seller_sku_code'] == sku][['order_date', 'quantity_sold']]
    sku_data = sku_data.set_index('order_date')
    
    # Ensure the index is a proper time series
    sku_data = sku_data.asfreq('D')  # Daily frequency
    sku_data = sku_data.fillna(0)  # Fill missing values with 0 (no sales)

    # Split the data into training and testing sets
    train = sku_data.iloc[:-1]  # Use all but the last day for training
    test = sku_data.iloc[-1:]  # Use the last day for testing
    
    # Fit SARIMAX model
    try:
        model = SARIMAX(
            train['quantity_sold'], 
            order=(1, 1, 1),  # p, d, q
            seasonal_order=(1, 1, 1, 7),  # P, D, Q, s (weekly seasonality)
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        
        # Forecast the next day
        forecast = result.forecast(steps=1)
        forecast_value = forecast.iloc[0]
        
        # Append the forecast result
        all_forecasts.append({
            'seller_sku_code': sku,
            'forecast_date': test.index[0] + pd.Timedelta(days=1),
            'forecasted_demand': forecast_value
        })
        
    except Exception as e:
        print(f"Error processing SKU {sku}: {e}")

# Convert forecasts to a DataFrame
forecast_df = pd.DataFrame(all_forecasts)

# Display forecasts
print(forecast_df)

# Save the forecast to a CSV file
forecast_df.to_csv("sarimax_demand_forecast.csv", index=False)
```

---

### Explanation:

1. **SARIMAX Parameters:**
   - `order=(p, d, q)`: Non-seasonal ARIMA parameters.
   - `seasonal_order=(P, D, Q, s)`: Seasonal ARIMA parameters (e.g., weekly seasonality, `s=7`).
   - These parameters can be tuned using techniques like grid search.

2. **Handling Missing Dates:**
   - The dataset is converted into a time series with daily frequency using `.asfreq('D')`.
   - Missing dates are filled with zeros, assuming no sales on those days.

3. **Forecasting:**
   - For each SKU, the model is trained on all but the last day, and the last day is used to forecast the next day’s demand.

4. **Output:**
   - A DataFrame (`forecast_df`) contains the forecasted demand for the next day for each SKU.
   - Saved as `sarimax_demand_forecast.csv`.

---

### Sample Output:

The output (`sarimax_demand_forecast.csv`) will look like this:

| seller_sku_code | forecast_date | forecasted_demand |
|------------------|--------------|-------------------|
| SKU-1           | 2024-12-02   | 10.45             |
| SKU-2           | 2024-12-02   | 8.12              |

---

### Notes:
- **Tuning Parameters:** Adjust `(p, d, q)` and `(P, D, Q, s)` for better results.
- **Error Handling:** The `try-except` block ensures the loop continues even if one SKU fails to train.
- **Evaluation:** You can add RMSE calculation by comparing forecasts with actuals (if available).

This approach works well for datasets with clear trends and seasonality. However, for sparse data, consider alternative models like LightGBM or Prophet.