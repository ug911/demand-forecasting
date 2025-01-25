import pandas as pd
import numpy as np
import scipy.stats as stats

# Load data
sku_code_column = 'masked_sku_code'
quantity_column = 'quantity'
data = pd.read_csv("sku_data.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Initialize an empty DataFrame to store forecasts
all_forecasts = []

# Number of simulations
n_simulations = 1000

# Loop through each SKU
sku_codes = data[sku_code_column].unique()

for sku in sku_codes:
    # Filter data for the current SKU
    sku_data = data[data[sku_code_column] == sku][['order_date', quantity_column]]
    historical_demand = sku_data[quantity_column]

    # Fit a distribution (assume Normal for simplicity)
    mean = np.mean(historical_demand)
    std_dev = np.std(historical_demand)

    if std_dev == 0:  # Handle case where all demands are the same
        std_dev = 1e-6

    # Monte Carlo Simulation
    simulations = np.random.normal(loc=mean, scale=std_dev, size=n_simulations)
    simulations = np.clip(simulations, 0, None)  # Ensure demand is non-negative

    # Aggregate simulation results
    forecasted_demand = np.mean(simulations)  # Use mean as forecast

    # Store the forecast
    all_forecasts.append({
        sku_code_column: sku,
        'forecast_date': sku_data['order_date'].max() + pd.Timedelta(days=1),
        'forecasted_demand': forecasted_demand
    })

# Convert forecasts to a DataFrame
forecast_df = pd.DataFrame(all_forecasts)

# Display forecasts
print(forecast_df)

# Save the forecast to a CSV file
forecast_df.to_csv("forecasted_data/monte_carlo_demand_forecast.csv", index=False)