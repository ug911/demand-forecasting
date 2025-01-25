import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

sku_code_column = 'masked_sku_code'
quantity_column = 'quantity'

# Load data
data = pd.read_csv("sample_data/masked_sku_codes.csv")
data['order_date'] = pd.to_datetime(data['order_date'])
data = data.sort_values(by=[sku_code_column, 'order_date'])

# Initialize a list to store forecasts with confidence intervals
forecasts_with_ci = []

# Loop through each SKU
sku_codes = data[sku_code_column].unique()
count = 1
for sku in sku_codes:

    # Filter data for the current SKU
    sku_data = data[data[sku_code_column] == sku][['order_date', quantity_column]]
    sku_data = sku_data.set_index('order_date')
    sku_data = sku_data.asfreq('D').fillna(0)  # Fill missing dates with zeros

    print(count, sku, len(sku_data))
    # Split into train data (all except last day)
    train = sku_data.iloc[:-1]

    try:
        # Fit SARIMAX model
        model = SARIMAX(
            train[quantity_column],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),  # Weekly seasonality
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        result = model.fit(disp=False)

        # Forecast next day with confidence intervals
        forecast_result = result.get_forecast(steps=1)
        forecast_mean = forecast_result.predicted_mean.iloc[0]
        conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
        lower_bound = conf_int.iloc[0, 0]
        upper_bound = conf_int.iloc[0, 1]

        # Store the result
        forecasts_with_ci.append({
            sku_code_column : sku,
            'forecast_date': train.index[-1] + pd.Timedelta(days=1),
            'forecasted_demand': forecast_mean,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })
    except Exception as e:
        print(e)
    count += 1

# Convert forecasts to a DataFrame
forecast_df = pd.DataFrame(forecasts_with_ci)

# Display the result
print(forecast_df)

# Save the result to a CSV file
forecast_df.to_csv("forecasted_data/demand_forecast_with_ci.csv", index=False)