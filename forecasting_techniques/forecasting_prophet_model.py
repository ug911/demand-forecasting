import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

sku_code_column = 'masked_sku_code'
quantity_column = 'quantity'

def plot_forecast_for_sku(data, sku_code):
    # Optional: Plot the forecast for a specific SKU
    sku_to_plot = sku_codes  # Replace with desired SKU
    sku_data = data[data[sku_code_column] == sku_to_plot][['order_date', quantity_column]]
    sku_data = sku_data.rename(columns={'order_date': 'ds', quantity_column: 'y'})
    model = Prophet()
    model.fit(sku_data)
    future = model.make_future_dataframe(periods=7)  # Forecast for 7 days
    forecast = model.predict(future)
    model.plot(forecast)
    plt.title(f"Forecast for {sku_to_plot}")
    plt.show()

# Load data
data = pd.read_csv("sample_data/masked_sku_codes.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Initialize an empty DataFrame to store forecasts
all_forecasts = []

# Loop through each SKU
sku_codes = data[sku_code_column].unique()
count = 1
for sku in sku_codes:
    # Filter data for the current SKU
    sku_data = data[data[sku_code_column] == sku][['order_date', quantity_column]]
    sku_data = sku_data.set_index('order_date')
    print(sku_data)
    sku_data = sku_data.asfreq('D').fillna(0)
    print(sku_data)
    sku_data = sku_data.reset_index()



    # Rename columns to match Prophet requirements
    sku_data = sku_data.rename(columns={'order_date': 'ds', quantity_column: 'y'})

    # Initialize Prophet model
    model = Prophet()

    # Fit the model
    model.fit(sku_data)

    # Create a DataFrame for future dates (next day's forecast)
    future = model.make_future_dataframe(periods=7)  # Forecast for 1 day
    forecast = model.predict(future)

    # Store the forecasted value for the next day
    forecast_next_day = forecast.iloc[-7:][['ds', 'yhat']]
    forecast_next_day[sku_code_column] = sku  # Add SKU identifier

    print(count, sku, len(sku_data), forecast_next_day['yhat'])
    print(forecast_next_day)
    # Append to all_forecasts
    all_forecasts.append(forecast_next_day)
    count += 1

# Combine all forecasts
all_forecasts = pd.DataFrame(all_forecasts)

# Display the next day's demand forecast
print(all_forecasts)

# Save the forecast to a CSV
all_forecasts.to_csv("forecasted_data/prophet_demand_forecast_next_day.csv", index=False)


