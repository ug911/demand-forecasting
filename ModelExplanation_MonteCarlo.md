# Explaining Monte Carlo Simulations

Monte Carlo simulation is a probabilistic method that simulates multiple random scenarios to estimate future outcomes. For demand forecasting, we can simulate future sales based on historical demand distributions. Here’s how to implement Monte Carlo simulation for the given dataset:

---

### Steps:
1. **Data Preparation:** Extract historical demand for each SKU.
2. **Determine Distribution:** Fit a probability distribution (e.g., Normal, Poisson) to the historical sales data for each SKU.
3. **Simulate Demand:** Generate multiple random samples (simulations) from the fitted distribution for the next day.
4. **Aggregate Results:** Use the mean or median of the simulations as the forecasted demand.

---

### Code:

```python
import pandas as pd
import numpy as np
import scipy.stats as stats

# Load data
data = pd.read_csv("sku_data.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Initialize an empty DataFrame to store forecasts
all_forecasts = []

# Number of simulations
n_simulations = 1000

# Loop through each SKU
sku_codes = data['seller_sku_code'].unique()

for sku in sku_codes:
    # Filter data for the current SKU
    sku_data = data[data['seller_sku_code'] == sku][['order_date', 'quantity_sold']]
    historical_demand = sku_data['quantity_sold']
    
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
        'seller_sku_code': sku,
        'forecast_date': sku_data['order_date'].max() + pd.Timedelta(days=1),
        'forecasted_demand': forecasted_demand
    })

# Convert forecasts to a DataFrame
forecast_df = pd.DataFrame(all_forecasts)

# Display forecasts
print(forecast_df)

# Save the forecast to a CSV file
forecast_df.to_csv("monte_carlo_demand_forecast.csv", index=False)
```

---

### Explanation:

1. **Monte Carlo Simulation:**
   - For each SKU, historical demand is analyzed, and a probability distribution (Normal in this case) is assumed.
   - The `mean` and `std_dev` of the historical demand are used as parameters to simulate future scenarios.

2. **Simulations:**
   - `np.random.normal` generates `n_simulations` random values based on the Normal distribution.
   - `np.clip` ensures that no negative demand values are produced.

3. **Forecast Aggregation:**
   - The average (`mean`) of all simulated values is taken as the forecasted demand.

4. **Output:**
   - The `forecast_df` DataFrame contains the next day’s forecast for each SKU.

---

### Sample Output:

The output (`monte_carlo_demand_forecast.csv`) will look like this:

| seller_sku_code | forecast_date | forecasted_demand |
|------------------|--------------|-------------------|
| SKU-1           | 2024-12-02   | 12.34             |
| SKU-2           | 2024-12-02   | 8.56              |

---

### Notes:
1. **Adjust Distribution:**
   - If the sales data is not normally distributed, you can use other distributions (e.g., Poisson, Gamma). Replace `np.random.normal` with the appropriate sampling function.
   - Use statistical tests (e.g., Kolmogorov-Smirnov) to identify the best-fitting distribution.

2. **n_simulations:**
   - Increase `n_simulations` for more robust estimates at the cost of computation time.

3. **Simplicity:**
   - This method assumes independent daily demand. For complex dependencies (e.g., seasonality), consider integrating more advanced statistical models.