import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

sku_code_column = 'masked_sku_code'
quantity_column = 'quantity'

# Load data
data = pd.read_csv("sku_data.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Sort the data
data = data.sort_values(by=[sku_code_column, 'order_date'])

# Initialize an empty list to store forecasts
all_forecasts = []

# Loop through each SKU
sku_codes = data[sku_code_column].unique()

for sku in sku_codes:
    # Filter data for the current SKU
    sku_data = data[data[sku_code_column] == sku][['order_date', quantity_column]]
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
            train[quantity_column],
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
            sku_code_column: sku,
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
forecast_df.to_csv("forecasted_data/sarimax_demand_forecast.csv", index=False)
