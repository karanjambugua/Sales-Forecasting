import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load your data (adjust path accordingly)
data = pd.read_csv('data/Thrift_Company_Sales_Clean.csv')

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
monthly_data = data.resample('M').sum()
monthly_data['Month'] = monthly_data.index.month
monthly_data['Year'] = monthly_data.index.year
monthly_data['Prev_Sales'] = monthly_data['Amount'].shift(1)
monthly_data['Day_of_week'] = monthly_data.index.dayofweek
monthly_data.dropna(inplace=True)

# Prepare features and target variable
X = monthly_data[['Month', 'Prev_Sales', 'Day_of_week']]
y = monthly_data['Amount']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

### Step 1: Train Random Forest Model with default parameters
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation for Random Forest
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)

print("Random Forest - MSE:", rf_mse)
print("Random Forest - RMSE:", rf_rmse)

### Step 2: Train XGBoost Model with default parameters
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions and evaluation for XGBoost
xgb_predictions = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_rmse = np.sqrt(xgb_mse)

print("XGBoost - MSE:", xgb_mse)
print("XGBoost - RMSE:", xgb_rmse)

### Step 3: Future Sales Predictions (next 3 months)
# Use the last known sales data to predict future months (12, 1, 2)
future_months = pd.DataFrame({
    'Month': [12, 1, 2],  # Placeholder months
    'Prev_Sales': [monthly_data['Amount'].iloc[-1]] * 3,  # Use the last known sales as the previous sales
    'Day_of_week': [0, 1, 2]  # Use a dummy day of the week for the prediction (Monday, Tuesday, Wednesday)
})

# Random Forest Future Predictions
rf_future_predictions = rf_model.predict(future_months)
# XGBoost Future Predictions
xgb_future_predictions = xgb_model.predict(future_months)

print("Future Sales Predictions (Random Forest):", rf_future_predictions)
print("Future Sales Predictions (XGBoost):", xgb_future_predictions)
