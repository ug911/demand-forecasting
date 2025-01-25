import json
import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


class DemandForecaster:
    def __init__(self, sku_code, data):
        self.sku_code = sku_code
        self.data = data[data['masked_sku_codes'] == sku_code].copy()
        if self.data.empty:
            raise ValueError(f"No data found for SKU: {sku_code}")
        self.data['order_date'] = pd.to_datetime(self.data['order_date'])
        self.data = self.data.set_index('order_date')

    def fill_missing_dates(self, end_date):
        start_date = self.data.index.min()
        date_range = pd.date_range(start=start_date, end=end_date)
        filled_df = self.data.reindex(date_range)
        filled_df['quantity'] = filled_df['quantity'].fillna(0).astype(int)
        filled_df['masked_sku_codes'] = self.sku_code
        self.data = filled_df.reset_index().rename(columns={"index": "order_date"}).set_index("order_date")
        return self.data.copy()

    def _create_features(self, df):
        df = df.copy()
        df['quantity_log'] = np.log1p(df['quantity'])
        df['diff_1'] = df['quantity'].diff()
        df['diff_7'] = df['quantity'].diff(7)
        for lag in [1, 7, 30]:
            df[f'quantity_sold_lag_{lag}'] = df['quantity'].shift(lag)
        for window in [7, 30]:
            df[f'quantity_sold_avg_{window}'] = df['quantity'].rolling(window=window).mean().shift(1)
            df[f'sales_rolling_sum_{window}'] = df['quantity'].rolling(window=window).sum().shift(1)
            df[f'sales_rolling_std_{window}'] = df['quantity'].rolling(window=window).std().shift(1)
            df[f'sales_rolling_max_{window}'] = df['quantity'].rolling(window=window).max().shift(1)
            df[f'sales_rolling_min_{window}'] = df['quantity'].rolling(window=window).min().shift(1)
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['year'] = df.index.year
        for period in [7, 30, 365]:
            df[f'sin_{period}'] = np.sin(2 * np.pi * df['day_of_year'] / period)
            df[f'cos_{period}'] = np.cos(2 * np.pi * df['day_of_year'] / period)
        try:
            decomposition = seasonal_decompose(df['quantity'].fillna(0), model='additive', period=7,
                                               extrapolate_trend='freq')
            df['trend'] = decomposition.trend
            df['seasonal'] = decomposition.seasonal
            df['residual'] = decomposition.resid
        except:
            pass

        df['weighted_avg'] = 0.0
        alpha = 0.3
        for i in range(1, len(df)):
            df['weighted_avg'].iloc[i] = alpha * df['quantity'].iloc[i - 1] + (1 - alpha) * df['weighted_avg'].iloc[
                i - 1]

        df['quantity_sold_lag_1_x_day_of_week'] = df['quantity_sold_lag_1'] * df['day_of_week']
        df['quantity_sold_lag_7_x_day_of_week'] = df['quantity_sold_lag_7'] * df['day_of_week']
        df['quantity_sold_lag_30_x_day_of_week'] = df['quantity_sold_lag_30'] * df['day_of_week']

        # Anomaly Detection
        try:
            scaler = StandardScaler()
            scaled_quantity = scaler.fit_transform(df[['quantity']].fillna(0))
            model = IsolationForest(contamination=0.05)  # Adjust contamination as needed
            df['is_anomaly'] = model.fit_predict(scaled_quantity)
            df['is_anomaly'] = df['is_anomaly'].map({-1: 1, 1: 0})
        except:
            pass

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill')
        df = df.fillna(method='bfill')
        return df

    def _evaluate_forecast(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse

    def moving_average(self, n):
        forecasts = self.data['quantity'].rolling(window=n).mean().shift(1)
        return forecasts

    def linear_regression(self, n):
        df = self.data.copy()
        df['t'] = range(len(df))
        X = df['t'].values.reshape(-1, 1)
        y = df['quantity'].values
        model = LinearRegression()
        model.fit(X[:-n], y[:-n])
        forecasts = model.predict(np.array(range(len(df), len(df) + n)).reshape(-1, 1))
        return pd.Series(forecasts, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n))

    def exponential_smoothing(self, n):
        model = ExponentialSmoothing(self.data['quantity'], seasonal=None, initialization_method="estimated").fit()
        forecasts = model.forecast(n)
        return forecasts

    def sarimax(self, n, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        try:
            model = SARIMAX(self.data['quantity'], order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False).fit()
            forecasts = model.forecast(n)
            return forecasts
        except:
            return None

    def xgboost(self, n):
        df = self._create_features(self.data)
        X = df.drop('quantity', axis=1)
        y = df['quantity']
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X[:-n], y[:-n])
        forecast_X = X[-n:]
        forecasts = model.predict(forecast_X)
        return pd.Series(forecasts, index=forecast_X.index)

        # df['t'] = range(len(df))
        # X = df[['t']]
        # y = df['quantity']
        # model = xgb.XGBRegressor()
        # model.fit(X[:-n], y[:-n])
        # forecasts = model.predict(np.array(range(len(df), len(df) + n)).reshape(-1, 1))
        # return pd.Series(forecasts, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n))

    def lightgbm(self, n):
        df = self._create_features(self.data)
        X = df.drop('quantity', axis=1)
        y = df['quantity']
        model = lgb.LGBMRegressor()
        model.fit(X[:-n], y[:-n])
        forecast_X = X[-n:]
        forecasts = model.predict(forecast_X)
        return pd.Series(forecasts, index=forecast_X.index)

        # df = self.data.copy()
        # df['t'] = range(len(df))
        # X = df[['t']]
        # y = df['quantity']
        # model = lgb.LGBMRegressor()
        # model.fit(X[:-n], y[:-n])
        # forecasts = model.predict(np.array(range(len(df), len(df) + n)).reshape(-1, 1))
        # return pd.Series(forecasts, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n))

    def prophet(self, n):
        df = self.data.copy()
        df = df.reset_index().rename(columns={'order_date': 'ds', 'quantity': 'y'})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=n)
        forecast = model.predict(future)
        return forecast['yhat'][-n:].values

    def croston_method(self, n):
        y = self.data['quantity'].values
        demand = y > 0
        intervals = np.diff(np.nonzero(demand)[0])
        if len(intervals) == 0:
            intervals = np.array([np.inf])
        alpha = 0.1
        beta = 0.1
        level = np.mean(y[demand]) if np.any(demand) else 0
        intermittent = np.mean(intervals) if np.any(intervals != np.inf) else np.inf

        forecasts = []
        for _ in range(n):
            level = alpha * y[-1] + (1 - alpha) * level if y[-1] > 0 else (1 - alpha) * level
            intermittent = beta * intervals[-1] + (1 - beta) * intermittent if len(intervals) > 0 and intervals[
                -1] != np.inf else (1 - beta) * intermittent
            forecast = level / intermittent if intermittent != np.inf else 0
            forecasts.append(forecast)
            y = np.append(y, 0)
            intervals = np.append(intervals, np.inf)

        return np.array(forecasts)

    def tscb_method(self, n):
        y = self.data['quantity'].values
        demand = y > 0
        intervals = np.diff(np.nonzero(demand)[0])
        if len(intervals) == 0:
            intervals = np.array([np.inf])
        alpha = 0.1
        beta = 0.1
        level = np.mean(y[demand]) if np.any(demand) else 0
        intermittent = np.mean(intervals) if np.any(intervals != np.inf) else np.inf

        forecasts = []
        for _ in range(n):
            level = alpha * y[-1] + (1 - alpha) * level if y[-1] > 0 else (1 - alpha) * level
            intermittent = beta * intervals[-1] + (1 - beta) * intermittent if len(intervals) > 0 and intervals[
                -1] != np.inf else (1 - beta) * intermittent
            forecast = level * (1 - beta + beta / intermittent) if intermittent != np.inf else 0
            forecasts.append(forecast)
            y = np.append(y, 0)
            intervals = np.append(intervals, np.inf)
        return np.array(forecasts)

    def moving_average_with_confidence(self, n, confidence=0.95):
        forecasts = self.data['quantity'].rolling(window=n).mean().shift(1)
        # For MA, a simple confidence interval isn't directly applicable in the same way as for regression models.
        # This is a simplification and might not be statistically rigorous.
        std_err = self.data['quantity'].rolling(window=n).std().shift(1)
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        lower = forecasts - z * (std_err / np.sqrt(n))
        upper = forecasts + z * (std_err / np.sqrt(n))

        return forecasts, lower, upper

    def linear_regression_with_confidence(self, n, confidence=0.95):
        df = self.data.copy()
        df['t'] = range(len(df))
        X = df['t'].values.reshape(-1, 1)
        y = df['quantity'].values
        model = LinearRegression()
        model.fit(X[:-n], y[:-n])
        forecast_X = np.array(range(len(df), len(df) + n)).reshape(-1, 1)
        forecasts = model.predict(forecast_X)

        # Calculate confidence intervals
        predictions = model.predict(X[:-n])
        residuals = y[:-n] - predictions
        std_err = np.std(residuals)
        t = stats.t.ppf((1 + confidence) / 2, len(y[:-n]) - 2)  # Degrees of freedom = n - 2
        margin_of_error = t * std_err * np.sqrt(
            1 + 1 / len(y[:-n]) + (forecast_X - np.mean(X[:-n])) ** 2 / np.sum((X[:-n] - np.mean(X[:-n])) ** 2))
        lower = forecasts - margin_of_error.flatten()
        upper = forecasts + margin_of_error.flatten()
        return (
            pd.Series( forecasts, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n)),
            pd.Series( lower, index=pd.date_range(start=df.index[-1] + pd.Timedelta( days=1), periods=n)),
            pd.Series( upper, index=pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n))
        )

    def exponential_smoothing_with_confidence(self, n, confidence=0.95):
        model = ExponentialSmoothing(self.data['quantity'], seasonal=None, initialization_method="estimated").fit()
        forecasts = model.forecast(n)
        forecasts_series = pd.Series(forecasts, index=pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=n))
        try:
            pred_int = model.get_prediction(start=len(self.data), end=len(self.data) + n - 1).conf_int(alpha=1 - confidence)
            lower = pd.Series(pred_int[:, 0], index=pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=n))
            upper = pd.Series(pred_int[:, 1], index=pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=n))
        except:
            lower = None
            upper = None
        return forecasts_series, lower, upper

    def sarimax_with_confidence(self, n, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), confidence=0.95):
        try:
            model = SARIMAX(self.data['quantity'], order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False).fit()
            forecasts = model.get_forecast(steps=n)
            mean_forecast = forecasts.predicted_mean
            conf_int = forecasts.conf_int(alpha=1 - confidence)

            forecasts_series = pd.Series(mean_forecast, index=pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=n))
            lower = pd.Series(conf_int[:, 0], index=pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=n))
            upper = pd.Series(conf_int[:, 1], index=pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=n))
            return forecasts_series, lower, upper
        except:
            return None, None, None


# Example usage (replace with your actual data)
data = {'order_date': ['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-12', '2023-01-20', '2023-01-22'],
        'quantity': [5, 10, 0, 20, 5, 10],
        'masked_sku_codes': ['SKU123', 'SKU123', 'SKU123', 'SKU123', 'SKU123', 'SKU123']}
df = pd.DataFrame(data)

forecaster = DemandForecaster('SKU123', df)
filled_data = forecaster.fill_missing_dates(datetime.date(2023, 2, 28))
print("Filled Data")
print(filled_data)