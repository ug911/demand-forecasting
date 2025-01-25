# Feature Engineering

When training a **LightGBM** or **XGBoost** model for a time series of SKU order quantities, feature engineering is crucial. Unlike autoregressive models (e.g., ARIMA), these models do not natively handle time dependence, so you must explicitly create features to capture temporal patterns, seasonality, and SKU-specific characteristics.

### Important Considerations & Practical Tips
1. **Feature Importance:** After training the model, use feature importance plots to identify the most impactful features.
   
2. **Handling Missing Values:** Fill missing values in lag or rolling features appropriately (e.g., with zero or mean).

3. **Time Window:** Choose an appropriate time window for training (e.g., last 12 months) to balance recency and data size.

4. **Scaling:** Use standard scaling or normalization for features if required by the model.

5. **Feature Selection:** Use techniques like SHAP, permutation importance, or feature selection algorithms to pick the most relevant features.

6. **Cross-Validation:** Use time-series split for validation to avoid data leakage.

### Key Features for SKU Time Series Forecasting

#### 1. **Lag Features (Autoregressive Features)**
Lag features help capture temporal dependencies in the data:
- Quantities sold on previous days:
  - `quantity_sold_lag_1`: Quantity sold 1 day ago.
  - `quantity_sold_lag_7`: Quantity sold 7 days ago (weekly seasonality).
  - `quantity_sold_lag_30`: Quantity sold 30 days ago (monthly seasonality).
- Moving averages:
  - `quantity_sold_avg_7`: Average sales in the last 7 days.
  - `quantity_sold_avg_30`: Average sales in the last 30 days.
- Exponentially weighted moving averages:
  - Captures trends while giving more weight to recent data.

---

#### 2. **Date-Based Features**
These capture seasonality and temporal patterns:
- **Day/Month Information:**
  - `day_of_week`: Day of the week (0=Monday, 6=Sunday).
  - `is_weekend`: 1 if the day is Saturday/Sunday, 0 otherwise.
  - `day_of_month`: Day of the month (1-31).
  - `month`: Month (1-12).
  - `quarter`: Quarter of the year (1-4).
  - `is_holiday`: 1 if the day is a holiday, 0 otherwise (use a holiday calendar API or library).
- **Time Since Events:**
  - `days_since_last_sale`: Days since the last non-zero sales event.
  - `days_since_promotion`: Days since a promotion or event.

---

#### 3. **Rolling Statistics**
Rolling windows smooth variability and capture short-term trends:
- Sales in the last \( n \) days:
  - `sales_rolling_sum_7`: Total sales in the last 7 days.
  - `sales_rolling_sum_30`: Total sales in the last 30 days.
- Rolling standard deviation:
  - `sales_rolling_std_7`: Variability of sales in the last 7 days.
- Rolling max/min:
  - `sales_rolling_max_7`: Maximum sales in the last 7 days.
  - `sales_rolling_min_7`: Minimum sales in the last 7 days.

---

#### 4. **SKU-Level Features**
These are SKU-specific attributes that influence demand:
- **Category Information:**
  - `sku_category`: Product category or type (e.g., electronics, clothing).
  - `sku_price`: Current price of the SKU.
  - `discount`: Discount percentage (if applicable).
- **Historical Aggregates:**
  - `avg_quantity_sold`: Average quantity sold historically.
  - `max_quantity_sold`: Maximum quantity sold historically.
  - `total_quantity_sold`: Total quantity sold historically.

---

#### 5. **Event Features**
Include external events that impact demand:
- **Promotions or Discounts:**
  - `is_promotion`: 1 if there is a promotion for the SKU on that day.
- **Seasonality or Festivals:**
  - `is_festival`: 1 if the day is during a festival season.
  - `promotion_intensity`: Level of promotion (e.g., 0 for no promotion, 1-10 scale for discounts).

---

#### 6. **Weather Data**
Weather can impact sales for specific products:
- **Weather Conditions:**
  - `temperature`: Daily temperature in the sales region.
  - `rainfall`: Amount of rainfall.
  - `is_snow`: 1 if it is snowing, 0 otherwise.

---

#### 7. **Demand Trend Features**
Features that capture long-term demand changes:
- **Trend Indicators:**
  - `cumulative_sales`: Cumulative quantity sold over time.
  - `growth_rate`: Percentage change in sales compared to the previous period.

---

#### 8. **Target Transformations**
If the target variable has high variability, consider transformations:
- **Logarithmic Transformation:**
  - Apply a log transformation to stabilize variance.
- **Difference Transformation:**
  - Use the difference between consecutive days’ quantities (`diff_1`, `diff_7`).

---

### Example Feature Engineering Code in Python

```python
import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("sku_data.csv")
data['order_date'] = pd.to_datetime(data['order_date'])
data = data.sort_values(by=['seller_sku_code', 'order_date'])

# Feature Engineering
data['day_of_week'] = data['order_date'].dt.dayofweek
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
data['day_of_month'] = data['order_date'].dt.day
data['month'] = data['order_date'].dt.month
data['quarter'] = data['order_date'].dt.quarter

# Lag Features
for lag in [1, 7, 30]:
    data[f'quantity_sold_lag_{lag}'] = data.groupby('seller_sku_code')['quantity_sold'].shift(lag)

# Rolling Features
data['rolling_avg_7'] = data.groupby('seller_sku_code')['quantity_sold'].transform(lambda x: x.rolling(7).mean())
data['rolling_std_7'] = data.groupby('seller_sku_code')['quantity_sold'].transform(lambda x: x.rolling(7).std())

# Historical Aggregates
sku_stats = data.groupby('seller_sku_code')['quantity_sold'].agg(['mean', 'max', 'sum']).reset_index()
sku_stats.rename(columns={'mean': 'avg_quantity_sold', 'max': 'max_quantity_sold', 'sum': 'total_quantity_sold'}, inplace=True)
data = data.merge(sku_stats, on='seller_sku_code', how='left')

# Target Transformation (Optional)
data['log_quantity_sold'] = np.log1p(data['quantity_sold'])

# Drop rows with NaN values created by lagging
data = data.dropna()

# Save the processed data
data.to_csv("processed_sku_data.csv", index=False)
```

---


### Advanced features 
These features can help uncover hidden patterns and make the model even more predictive. Advanced features like these can significantly improve your model's ability to capture complex temporal patterns, seasonalities, and SKU-specific nuances in demand.

---

### 9. **Fourier Transform Features**
Fourier transforms can model seasonality and periodic trends effectively:
- Compute Fourier terms for time-based features:
  - `sin(2πt / T)` and `cos(2πt / T)` for different seasonal periods (daily, weekly, yearly).
  - For example, \( t \) could be the day of the year, and \( T \) the total number of days in a year.

```python
import numpy as np

T = 365  # yearly seasonality
data['fourier_sin'] = np.sin(2 * np.pi * data['order_date'].dt.dayofyear / T)
data['fourier_cos'] = np.cos(2 * np.pi * data['order_date'].dt.dayofyear / T)
```

---

### 10. **Decomposition Features**
Decompose the time series into **trend**, **seasonality**, and **residuals**:
- Use decomposition libraries like `statsmodels` or `seasonal_decompose` to extract these components.
- Add the decomposed components as features.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data['quantity_sold'], model='additive', period=7)
data['trend'] = result.trend
data['seasonality'] = result.seasonal
data['residual'] = result.resid
```

---

### 11. **Anomaly Features**
Detect and encode anomalies in the data, as anomalies often affect future demand:
- Use techniques like Z-score, Isolation Forest, or LOF (Local Outlier Factor).
- Create a binary `is_anomaly` feature.

```python
from scipy.stats import zscore

data['z_score'] = zscore(data['quantity_sold'])
data['is_anomaly'] = (data['z_score'] > 3).astype(int)
```

---

### 12. **Price Elasticity Features**
Demand often depends on price changes:
- **Price Differences:**
  - Difference between current and previous price.
  - `price_diff = current_price - previous_price`.
- **Elasticity Ratio:**
  - `price_elasticity = percent_change_in_quantity / percent_change_in_price`.

---

### 13. **Weather Interaction Terms**
Interaction terms combine weather and SKU data:
- `rainfall * is_seasonal_product`: Captures the effect of rainfall on seasonal SKUs.
- `temperature * quantity_sold_lag_1`: Interaction between lagged demand and temperature.

---

### 14. **Covariate Interaction Features**
Create interaction features between time-based, SKU-level, and historical features:
- `quantity_sold_lag_1 * day_of_week`: Captures lagged demand impact on specific weekdays.
- `rolling_avg_30 * is_holiday`: Captures long-term average demand during holidays.

---

### 15. **Dynamic Segmentation**
Segment SKUs dynamically based on performance or demand:
- High/Medium/Low demand segments.
- Create a binary `is_high_demand` flag based on sales thresholds.

```python
data['is_high_demand'] = (data['quantity_sold'] > data['quantity_sold'].quantile(0.75)).astype(int)
```

---

### 16. **Event Correlation Features**
Quantify the impact of specific events:
- Measure the correlation between events (e.g., promotions, holidays) and demand.
- Create binary or weighted event features (`event_impact_score`).

---

### 17. **Temporal Attention Features**
Instead of simple lags, add weights based on temporal importance:
- Use exponential smoothing to give more weight to recent data:
  - `weighted_avg_t = α * quantity_t + (1 - α) * weighted_avg_t-1`.

---

### 18. **Multi-Step Forecasting Features**
For multi-step forecasting, aggregate future values as targets:
- Create features for cumulative demand over the next 7 days:
  - `future_7d_demand = sum(quantity_sold_t+1 to quantity_sold_t+7)`.

---

### 19. **Stockout Features**
Stockouts can heavily influence demand patterns:
- **Stockout Indicators:**
  - `is_stockout = (current_inventory == 0).astype(int)`.
- **Stockout Impact:**
  - Measure demand drop in periods immediately after a stockout.

---

### 20. **Embedding Features**
Use embeddings to encode categorical features:
- For SKU codes, train embeddings using techniques like Word2Vec, FastText, or autoencoders.
- Add embedding vectors as features for SKUs.

---

### 21. **Cluster-Based Features**
Cluster SKUs based on their demand patterns:
- Use k-means or hierarchical clustering on historical sales.
- Assign cluster IDs to SKUs and use them as features.

---

### 22. **Long-Term Memory Features**
Aggregate demand over larger time periods:
- `sales_last_90_days`: Total sales in the last 90 days.
- `sales_std_last_180_days`: Variability over the past 6 months.

---

### 23. **Gradient-Boosted Statistical Features**
Let a separate LightGBM model calculate feature importance on lagged data, and include the top features in your main model.

---

### 24. **Calendar Distance Features**
Capture the time distance to key events:
- `days_to_next_holiday`: Days until the next holiday.
- `days_since_promotion_start`: Days since a promotion began.



With these features, LightGBM or XGBoost can effectively capture the patterns in your SKU demand data and provide accurate forecasts.