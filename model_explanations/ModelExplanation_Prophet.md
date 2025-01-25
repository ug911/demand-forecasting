# Explaining Prophet by Facebook

Here is the Python code to perform demand forecasting using **Facebook Prophet** for the given dataset. Prophet is especially suited for time series forecasting and can handle seasonality and trends effectively.

---

### Steps:
1. **Data Preparation:** Reformat the dataset for Prophet (requires columns named `ds` and `y`).
2. **Train Model for Each SKU:** Since Prophet forecasts one time series at a time, loop through each SKU.
3. **Forecast for the Next Day:** Predict demand for the next day for each SKU.

---

### Code:

```python
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("sku_data.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Initialize an empty DataFrame to store forecasts
all_forecasts = []

# Loop through each SKU
sku_codes = data['seller_sku_code'].unique()

for sku in sku_codes:
    # Filter data for the current SKU
    sku_data = data[data['seller_sku_code'] == sku][['order_date', 'quantity_sold']]
    
    # Rename columns to match Prophet requirements
    sku_data = sku_data.rename(columns={'order_date': 'ds', 'quantity_sold': 'y'})
    
    # Initialize Prophet model
    model = Prophet()
    
    # Fit the model
    model.fit(sku_data)
    
    # Create a DataFrame for future dates (next day's forecast)
    future = model.make_future_dataframe(periods=1)  # Forecast for 1 day
    forecast = model.predict(future)
    
    # Store the forecasted value for the next day
    forecast_next_day = forecast.iloc[-1][['ds', 'yhat']]
    forecast_next_day['seller_sku_code'] = sku  # Add SKU identifier
    
    # Append to all_forecasts
    all_forecasts.append(forecast_next_day)

# Combine all forecasts
all_forecasts = pd.DataFrame(all_forecasts)

# Display the next day's demand forecast
print(all_forecasts)

# Save the forecast to a CSV
all_forecasts.to_csv("demand_forecast_next_day.csv", index=False)

# Optional: Plot the forecast for a specific SKU
sku_to_plot = sku_codes[0]  # Replace with desired SKU
sku_data = data[data['seller_sku_code'] == sku_to_plot][['order_date', 'quantity_sold']]
sku_data = sku_data.rename(columns={'order_date': 'ds', 'quantity_sold': 'y'})
model = Prophet()
model.fit(sku_data)
future = model.make_future_dataframe(periods=7)  # Forecast for 7 days
forecast = model.predict(future)
model.plot(forecast)
plt.title(f"Forecast for {sku_to_plot}")
plt.show()
```

---

### Explanation:

1. **Prophet Model Requirements:**
   - `ds`: The datetime column.
   - `y`: The target column (demand or `quantity_sold` in this case).

2. **Per SKU Forecasting:**
   - Since Prophet works with one time series at a time, we loop through each `seller_sku_code` and forecast separately.

3. **Future Predictions:**
   - `make_future_dataframe(periods=1)` generates a DataFrame for forecasting the next day's demand.
   - `yhat` provides the forecasted demand.

4. **Visualization:**
   - You can visualize the forecast for any SKU using Prophet’s built-in `plot` method.

5. **Output:**
   - `demand_forecast_next_day.csv` will contain the forecasted demand (`yhat`) for the next day for all SKUs.

---

### Sample Output:

If you run the code, the output (`demand_forecast_next_day.csv`) will look like this:

| ds         | yhat     | seller_sku_code |
|------------|----------|-----------------|
| 2024-12-02 | 12.34567 | SKU-1           |
| 2024-12-02 | 8.76543  | SKU-2           |

This forecasted demand (`yhat`) is for **2024-12-02**, assuming the latest date in the dataset is **2024-12-01**.