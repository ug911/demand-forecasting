# Explaining Gradiant Boosting models

Below is a Python code snippet to train a gradient boosting model (using LightGBM or XGBoost) to predict the next day's demand for each SKU based on the provided dataset:

### Steps:
1. **Data Preparation:** Convert the date column and create features like lagged sales, moving averages, etc.
2. **Feature Engineering:** Add SKU-specific features and date-related features.
3. **Model Training:** Use LightGBM or XGBoost for prediction.

Here’s the implementation:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv("sku_data.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Sort by SKU and date
data = data.sort_values(by=['seller_sku_code', 'order_date'])

# Create lagged features
def create_lag_features(df, lags):
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('seller_sku_code')['quantity_sold'].shift(lag)
    return df

# Create rolling features
def create_rolling_features(df, windows):
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('seller_sku_code')['quantity_sold'].shift(1).rolling(window).mean()
    return df

# Feature engineering
data = create_lag_features(data, lags=[1, 2, 3])
data = create_rolling_features(data, windows=[3, 7])

# Add date-related features
data['day_of_week'] = data['order_date'].dt.dayofweek
data['month'] = data['order_date'].dt.month

# Drop rows with NaN values (introduced by lagging and rolling features)
data = data.dropna()

# Define features and target
features = [col for col in data.columns if col not in ['seller_sku_code', 'order_date', 'quantity_sold']]
target = 'quantity_sold'

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target], test_size=0.2, random_state=42
)

# Train a LightGBM model
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, free_raw_data=False)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=1000,
    early_stopping_rounds=50,
    verbose_eval=100,
)

# Predictions and evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

# Save the model
model.save_model("demand_forecast_model.txt")

# Feature importance
importances = model.feature_importance()
feature_names = features
for feature, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance}")
```

### Key Notes:
1. **Lag and Rolling Features:** These features provide temporal information about past demand.
2. **Date Features:** Including `day_of_week` and `month` helps capture seasonality and periodic trends.
3. **LightGBM vs XGBoost:** You can swap LightGBM for XGBoost by changing the library and parameter syntax.
4. **Handling SKUs Separately:** The code groups by `seller_sku_code` to ensure SKU-specific trends are captured.

Save the data as `sku_data.csv` and execute the script. Modify the feature engineering or model parameters based on your specific dataset and business requirements.