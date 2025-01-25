# Explaining Confidence Intervals

To generate a high-confidence statistical forecast with **upper and lower confidence limits**, you can combine a statistical forecasting model with the computation of prediction intervals. Confidence limits are often derived based on the uncertainty of the model's predictions and the variability in the data.

Here’s a general approach:

---

### Steps to Generate Forecast with Confidence Limits:
1. **Fit a Forecasting Model:** Use a statistical model like SARIMAX, Prophet, or Exponential Smoothing.
2. **Generate Forecast:** Produce point forecasts for the desired time horizon.
3. **Estimate Prediction Uncertainty:** Calculate the standard error of the forecast.
4. **Calculate Confidence Limits:** Use the standard error to calculate upper and lower bounds using the z-score or t-statistic.

The confidence interval is calculated as:
\[
\text{Upper Bound} = \hat{y} + Z \cdot SE
\]
\[
\text{Lower Bound} = \hat{y} - Z \cdot SE
\]
Where:
- \(\hat{y}\): Forecasted value.
- \(SE\): Standard error of the forecast.
- \(Z\): z-score corresponding to the desired confidence level (e.g., 1.96 for 95%).

---

### Python Code Example Using SARIMAX:

Here’s an example using SARIMAX to forecast demand with confidence intervals:

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Load data
data = pd.read_csv("sku_data.csv")
data['order_date'] = pd.to_datetime(data['order_date'])
data = data.sort_values(by=['seller_sku_code', 'order_date'])

# Initialize a list to store forecasts with confidence intervals
forecasts_with_ci = []

# Loop through each SKU
sku_codes = data['seller_sku_code'].unique()
for sku in sku_codes:
    # Filter data for the current SKU
    sku_data = data[data['seller_sku_code'] == sku][['order_date', 'quantity_sold']]
    sku_data = sku_data.set_index('order_date')
    sku_data = sku_data.asfreq('D').fillna(0)  # Fill missing dates with zeros

    # Split into train data (all except last day)
    train = sku_data.iloc[:-1]

    # Fit SARIMAX model
    model = SARIMAX(
        train['quantity_sold'],
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
        'seller_sku_code': sku,
        'forecast_date': train.index[-1] + pd.Timedelta(days=1),
        'forecasted_demand': forecast_mean,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    })

# Convert forecasts to a DataFrame
forecast_df = pd.DataFrame(forecasts_with_ci)

# Display the result
print(forecast_df)

# Save the result to a CSV file
forecast_df.to_csv("demand_forecast_with_ci.csv", index=False)
```

---

### Explanation of Key Components:

1. **`get_forecast` Method:**
   - The `get_forecast` method provides the forecasted value and its confidence intervals directly.

2. **Confidence Intervals:**
   - Adjust the `alpha` parameter in `conf_int(alpha=0.05)` to control the confidence level (e.g., `alpha=0.10` for 90%).

3. **Result Columns:**
   - `forecasted_demand`: Point estimate of the forecast.
   - `lower_bound` and `upper_bound`: 95% confidence limits.

---

### Example Output:

| seller_sku_code | forecast_date | forecasted_demand | lower_bound | upper_bound |
|------------------|--------------|-------------------|-------------|-------------|
| SKU-1           | 2024-12-02   | 12.3              | 10.5        | 14.1        |
| SKU-2           | 2024-12-02   | 8.1               | 6.7         | 9.5         |

---

### Notes:
1. **Confidence Width:**
   - Wider confidence intervals indicate higher uncertainty in the forecast.
   - Narrow confidence intervals are preferred but may not always be feasible for volatile data.

2. **Model Selection:**
   - The method works with any model that provides error metrics, including Prophet and LightGBM.

3. **Validation:**
   - Use historical data to validate the accuracy of your confidence intervals.

By following this approach, you can provide high-confidence statistical forecasts with reliable upper and lower limits for decision-making.