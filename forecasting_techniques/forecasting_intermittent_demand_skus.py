import numpy as np
import pandas as pd


def croston_method(data, alpha=0.1):
    """
    Croston's Method for intermittent demand forecasting.

    Parameters:
    - data: array-like, historical demand (zeros for no demand days).
    - alpha: smoothing parameter (0 < alpha ≤ 1).

    Returns:
    - forecast: Forecasted demand rate.
    """
    # Initialize variables
    n = len(data)
    demand = np.zeros(n)
    intervals = np.zeros(n)
    forecasts = np.zeros(n)

    # Initialize demand and interval smoothing
    for i in range(n):
        if data[i] > 0:
            demand[i] = data[i]
            intervals[i] = 1
            break

    for t in range(i + 1, n):
        if data[t] > 0:
            # Update demand and interval using exponential smoothing
            demand[t] = alpha * data[t] + (1 - alpha) * demand[t - 1]
            intervals[t] = alpha * (t - np.where(data[:t] > 0)[0][-1]) + (1 - alpha) * intervals[t - 1]
            forecasts[t] = demand[t] / intervals[t]
        else:
            demand[t] = demand[t - 1]
            intervals[t] = intervals[t - 1]
            forecasts[t] = forecasts[t - 1]

    return forecasts[-1]  # Return the last forecasted value


def tscb_method(data, alpha=0.1):
    """
    Teunter-Syntetos-Babai (TSCB) Method for intermittent demand forecasting.

    Parameters:
    - data: array-like, historical demand (zeros for no demand days).
    - alpha: smoothing parameter (0 < alpha ≤ 1).

    Returns:
    - forecast: Forecasted demand rate.
    """
    # Initialize variables
    n = len(data)
    demand = np.zeros(n)
    intervals = np.zeros(n)
    forecasts = np.zeros(n)

    # Initialize demand and interval smoothing
    for i in range(n):
        if data[i] > 0:
            demand[i] = data[i]
            intervals[i] = 1
            break

    for t in range(i + 1, n):
        if data[t] > 0:
            # Update demand and interval using exponential smoothing
            demand[t] = alpha * data[t] + (1 - alpha) * demand[t - 1]
            intervals[t] = alpha * (t - np.where(data[:t] > 0)[0][-1]) + (1 - alpha) * intervals[t - 1]
            # TSCB adjustment factor (1 - alpha / 2)
            forecasts[t] = (demand[t] / intervals[t]) * (1 - alpha / 2)
        else:
            demand[t] = demand[t - 1]
            intervals[t] = intervals[t - 1]
            forecasts[t] = forecasts[t - 1]

    return forecasts[-1]  # Return the last forecasted value


# Example Usage
if __name__ == "__main__":
    # Sample data: Intermittent demand (0s for no demand days)
    historical_demand = [0, 0, 3, 0, 0, 0, 5, 0, 2, 0, 0, 1, 0, 0]

    # Croston's Method Forecast
    croston_forecast = croston_method(historical_demand, alpha=0.1)
    print(f"Croston's Method Forecast: {croston_forecast:.2f}")

    # TSCB Method Forecast
    tscb_forecast = tscb_method(historical_demand, alpha=0.1)
    print(f"TSCB Method Forecast: {tscb_forecast:.2f}")