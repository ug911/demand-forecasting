import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load data
sku_code_column = 'masked_sku_code'
quantity_column = 'quantity'
data = pd.read_csv("sku_data.csv")

# Ensure order_date is a datetime object
data['order_date'] = pd.to_datetime(data['order_date'])

# Sort by SKU and date
data = data.sort_values(by=[sku_code_column, 'order_date'])

# Create lagged features
def create_lag_features(df, lags):
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(sku_code_column)[quantity_column].shift(lag)
    return df

# Create rolling features
def create_rolling_features(df, windows):
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby(sku_code_column)[quantity_column].shift(1).rolling(window).mean()
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
features = [col for col in data.columns if col not in [sku_code_column, 'order_date', quantity_column]]
target = quantity_column

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
model.save_model("gradiant_boosting_demand_forecast_model.txt")

# Feature importance
importances = model.feature_importance()
feature_names = features
for feature, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance}")
