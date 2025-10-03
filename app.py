from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
# The reasoning behind the outputs
# Load the sales data
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

# Function to check if input is seen by the model
def check_input_seen(input_data, column):
    """ Check if the input data exists in the column of training data """
    if input_data not in monthly_data[column].unique():
        return False
    return True

# Function to get default prediction
def get_default_prediction():
    """ Generate a default prediction using the historical average sales """
    avg_sales = monthly_data['Amount'].mean()
    return avg_sales

# Route for the user selection page (forecast selection)
@app.route('/user')
def user():
    # Fetch actual data (product names and shops) from your database or data file
    products = data['Product/Service'].unique() 
    shops = data['Shop'].unique()  
    
    # Calculate min and max quantity based on historical data (adjust this logic)
    min_quantity = int(data['Amount'].min())  # Convert to int
    max_quantity = int(data['Amount'].max())  # Convert to int

    # Render the user page and pass the products and shops to the template
    return render_template('user.html', products=products, shops=shops,
                           min_quantity=min_quantity, max_quantity=max_quantity)


# Route for Product Forecasting
# Route for Product Forecasting (train and predict using Random Forest)
@app.route('/predict-product', methods=['POST'])
def predict_product():
    # Extract product and quantity from the form
    product = request.form['product']
    quantity_range = request.form['quantity']
    
    # Split the quantity range (e.g., '0-999') into a start and end
    try:
        quantity_start, quantity_end = map(int, quantity_range.split('-'))
        # Use the middle of the range for prediction (you can change this logic if needed)
        quantity = (quantity_start + quantity_end) // 2
    except ValueError:
        return jsonify({
            'error': f"The quantity range '{quantity_range}' is not valid. Please use a valid range (e.g., '0-1000').",
            'recommendations': [
                "Please enter a valid quantity range for better predictions."
            ]
        })
    
    # Ensure the product is valid (exists in the dataset)
    if product not in data['Product/Service'].values:
        default_sales = get_default_prediction()
        return render_template('results.html', error=f"The product '{product}' is not recognized by the model. Returning default prediction.",
                               forecast=f"Predicted sales for {product}: {default_sales:.2f} units (default prediction).",
                               recommendations=[
                                   "Please provide more data for better predictions.",
                                   "Restock the product for the upcoming season."
                               ])

    
    # Ensure the quantity is within the valid range (based on the historical data)
    min_quantity = data['Amount'].min()
    max_quantity = data['Amount'].max()
    if quantity < min_quantity or quantity > max_quantity:
        return jsonify({
            'error': f"The quantity sold '{quantity}' is out of the valid range. Please enter a value between {min_quantity} and {max_quantity}.",
            'recommendations': [
                "Please provide a valid quantity for better predictions."
            ]
        })

    # Prepare the data for modeling
    X = monthly_data[['Month', 'Prev_Sales', 'Day_of_week']]
    y = monthly_data['Amount']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model (optional, for debugging purposes)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (MSE) for Product Forecasting: {mse}')

    # Generate forecast for the given quantity sold
    forecast_sales = model.predict([[1, quantity, 2]])  # Just an example, use actual data logic for predictions

    # Return the prediction results as a JSON response
    result = {
        'product': product,
        'quantity': quantity,
        'forecast': f"Predicted sales for {product}: {forecast_sales[0]:.2f} units.",
        'recommendations': [
            f"Restock {product} for the upcoming season.",
            f"Promote {product} during the holiday season."
        ]
    }

    return render_template('results.html', result=result)



# Route for Date-Based Forecasting
# Route for Date-Based Forecasting (using ARIMA or Random Forest)
@app.route('/predict-date', methods=['POST'])
def predict_date():
    start_date = request.form['startDate']
    end_date = request.form['endDate']

    # Example ARIMA model prediction for simplicity
    model_arima = ARIMA(monthly_data['Amount'], order=(5,1,0))
    model_arima_fit = model_arima.fit()

    forecast_arima = model_arima_fit.forecast(steps=3)

    result = {
        'start_date': start_date,
        'end_date': end_date,
        'forecast': f"Predicted sales from {start_date} to {end_date}: ${forecast_arima.sum():.2f} in total sales.",
        'recommendations': [
            "Increase stock for high-demand products.",
            "Offer discounts for slow-moving items."
        ]
    }

    return jsonify(result)

# Route for Shop Performance Forecasting
@app.route('/predict-shop', methods=['POST'])
def predict_shop():
    shop = request.form['shop']
    shop_sales = int(request.form['shopSales'])

    # Check if the shop is seen in the training data
    if not check_input_seen(shop, 'Shop'):
        default_sales = get_default_prediction()
        return jsonify({
            'error': f"The shop '{shop}' is not recognized by the model. Returning default prediction.",
            'forecast': f"Predicted sales for {shop}: {default_sales:.2f} in the next 3 months (default prediction based on historical data).",
            'recommendations': [
                "Please provide more data for better predictions.",
                "Increase stock of fast-moving products."
            ]
        })

    # Prepare the data for modeling
    X_shop = monthly_data[['Month', 'Prev_Sales', 'Day_of_week']]
    y_shop = monthly_data['Amount']

    # Train-Test Split
    X_train_shop, X_test_shop, y_train_shop, y_test_shop = train_test_split(X_shop, y_shop, test_size=0.2, shuffle=False)

    # Train the RandomForest Model
    model_shop = RandomForestRegressor(n_estimators=100)
    model_shop.fit(X_train_shop, y_train_shop)

    # Make predictions
    y_pred_shop = model_shop.predict(X_test_shop)

    # Evaluate the model
    mse_shop = mean_squared_error(y_test_shop, y_pred_shop)
    print(f'Mean Squared Error (MSE) for Shop Performance Forecasting: {mse_shop}')

    # Generate forecast
    forecast_shop = model_shop.predict([[1, shop_sales, 2]])  # Example

    result = {
        'shop': shop,
        'sales': shop_sales,
        'forecast': f"Predicted sales for {shop}: ${forecast_shop[0]:.2f} in the next 3 months.",
        'recommendations': [
            f"Increase stock of fast-moving products in {shop}.",
            f"Consider improving the layout to increase sales at {shop}."
        ]
    }

    return jsonify(result)

# Route for Day/Week/Weekend Forecasting
@app.route('/predict-dayweek', methods=['POST'])
def predict_dayweek():
    day_of_week = request.form['dayOfWeek']
    include_weekend = request.form['includeWeekend'] == 'on'

    # Check if the day_of_week is valid
    if day_of_week not in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        return jsonify({
            'error': f"The day '{day_of_week}' is not recognized. Please check the input."
        })

    # Prepare the data for modeling
    X_dayweek = monthly_data[['Month', 'Prev_Sales', 'Day_of_week']]
    y_dayweek = monthly_data['Amount']

    # Train-Test Split
    X_train_dayweek, X_test_dayweek, y_train_dayweek, y_test_dayweek = train_test_split(X_dayweek, y_dayweek, test_size=0.2, shuffle=False)

    # Train the RandomForest Model
    model_dayweek = RandomForestRegressor(n_estimators=100)
    model_dayweek.fit(X_train_dayweek, y_train_dayweek)

    # Make predictions
    y_pred_dayweek = model_dayweek.predict(X_test_dayweek)

    # Evaluate the model
    mse_dayweek = mean_squared_error(y_test_dayweek, y_pred_dayweek)
    print(f'Mean Squared Error (MSE) for Day/Week/Weekend Forecasting: {mse_dayweek}')

    # Generate forecast for a specific day (example)
    forecast_dayweek = model_dayweek.predict([[1, 1000, 5]])  # Example data point

    result = {
        'day_of_week': day_of_week,
        'include_weekend': include_weekend,
        'forecast': f"Predicted sales for {day_of_week}: ${forecast_dayweek[0]:.2f} in sales.",
        'recommendations': [
            "Consider offering promotions on weekends.",
            "Adjust stock levels based on predicted sales."
        ]
    }

    return jsonify(result)
# Route for the homepage (index page)
@app.route('/')
def index():
    return render_template('index.html')

# Route for the store manager page
@app.route('/store-manager')
def store_manager():
    return render_template('store_manager.html')

# Route for the procurement page
@app.route('/procurement')
def procurement():
    return render_template('procurement.html')

# Route for the executive page
@app.route('/executive')
def executive():
    return render_template('executive.html')

# Route to fetch sales forecast data (for frontend)
@app.route('/get-sales-forecast')
def get_sales_forecast():
    try:
        # Load your data from the data folder (replace with actual data path)
        data = pd.read_csv('data/Thrift_Company_Sales.csv')  # Ensure file path is correct
        if data.empty:
            raise ValueError("CSV file is empty")
        
        # Ensure 'Date' and 'Sales' columns exist
        if 'Date' not in data.columns or 'Sales' not in data.columns:
            raise ValueError("Required columns (Date, Sales) are missing")

        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Apply a simple forecasting model (Exponential Smoothing / Holt-Winters)
        model = ExponentialSmoothing(data['Sales'], trend='add', seasonal='add', seasonal_periods=12)
        fit = model.fit()

        # Forecast the next 3 months
        forecast_period = 3
        forecast = fit.forecast(forecast_period)
        
        # Calculate industry metrics
        industry_sales = data['Amount'].sum()  # Total sales
        forecast_accuracy = 94.2  # Mock value, replace with actual calculation if available
        active_alerts = 12  # Mock value
        inventory_turnover = 8.5  # Mock value

        # Preparing forecast data for frontend
        forecast_data = {
            'months': ['January', 'February', 'March'],
            'sales': forecast.tolist(),
            'industrySales': f"${industry_sales / 1e6:.2f}M",  # Format to millions
            'forecastAccuracy': f"{forecast_accuracy}%",
            'activeAlerts': active_alerts,
            'inventoryTurnover': f"{inventory_turnover}x",
            'bestSellingProducts': [
                {'name': 'Winter Jacket', 'forecastedSales': 500},
                {'name': 'Sneakers', 'forecastedSales': 300},
                {'name': 'Vintage T-shirt', 'forecastedSales': 250}
            ],
            'recommendations': [
                "Restock Winter Jackets for December",
                "Transfer sneakers to high-demand stores",
                "Offer promotions for vintage T-shirts"
            ]
        }

        # Return forecast data as JSON
        return jsonify(forecast_data)

    except Exception as e:
        # Log the exception and return error message
        print(f"Error in getting sales forecast: {e}")
        return jsonify({'error': str(e)}), 500

# Route to fetch store-specific sales forecast data (for frontend)
@app.route('/get-store-sales-forecast')
def get_store_sales_forecast():
    # Mock data for store manager (this can be replaced with actual data fetching and processing)
    store_forecast_data = {
        'months': ['January', 'February', 'March'],
        'sales': [5000, 5500, 6000],
        'recommendations': [
            "Restock Winter Jackets for high-demand areas",
            "Promote Sneakers through discounts",
            "Transfer Vintage T-shirts to new stores"
        ]
    }
    return jsonify(store_forecast_data)

# Route to fetch procurement-related data (for frontend)
@app.route('/get-procurement-data')
def get_procurement_data():
    # Mock data for procurement (this can be replaced with actual data processing)
    procurement_data = {
        'categories': ['Clothing', 'Electronics', 'Accessories'],
        'stockAndDemand': [3500, 2000, 1500],
        'replenishmentSuggestions': [
            "Order more Clothing for upcoming season",
            "Electronics need timely restocking",
            "Accessories should be reordered before the next sale"
        ]
    }
    return jsonify(procurement_data)

# Route to fetch executive-level sales data (for frontend)
@app.route('/get-executive-sales-data')
def get_executive_sales_data():
    # Mock data for executives (this can be replaced with actual data processing)
    executive_sales_data = {
        'months': ['January', 'February', 'March', 'April', 'May', 'June'],
        'sales': [75000, 80000, 78000, 85000, 90000, 92000]
    }
    return jsonify(executive_sales_data)
# Route to user.html

# Route for Day/Week/Weekend Forecasting form submission


# Placeholder functions for running the forecast models
# Replace these with your actual model code
def run_product_forecast_model(product, quantity):
    # Placeholder logic for product forecast model
    return {"product": product, "quantity": quantity, "forecast": f"Predicted sales for {product} based on {quantity} units sold."}

def run_date_forecast_model(start_date, end_date):
    # Placeholder logic for date-based forecast model
    return {"start_date": start_date, "end_date": end_date, "forecast": f"Predicted sales from {start_date} to {end_date}."}

def run_shop_forecast_model(shop, shop_sales):
    # Placeholder logic for shop forecast model
    return {"shop": shop, "sales": shop_sales, "forecast": f"Predicted sales for {shop} based on {shop_sales} sales."}

def run_dayweek_forecast_model(day_of_week, include_weekend):
    # Placeholder logic for day/week/ weekend forecast model
    return {"day_of_week": day_of_week, "include_weekend": include_weekend, "forecast": f"Predicted sales on {day_of_week} (Weekend: {include_weekend})."}

# Route to display the results page
@app.route('/results')
def results():
    forecast = request.args.get('forecast', type=float)
    error = request.args.get('error', type=str, default=None)
    product = request.args.get('product', type=str, default=None)
    quantity = request.args.get('quantity', type=int, default=None)
    start_date = request.args.get('start_date', type=str, default=None)
    end_date = request.args.get('end_date', type=str, default=None)
    shop = request.args.get('shop', type=str, default=None)
    shop_sales = request.args.get('shop_sales', type=int, default=None)
    day_of_week = request.args.get('day_of_week', type=str, default=None)
    include_weekend = request.args.get('include_weekend', type=str, default=None)

    return render_template('results.html', forecast=forecast, error=error, product=product, quantity=quantity,
                           start_date=start_date, end_date=end_date, shop=shop, shop_sales=shop_sales,
                           day_of_week=day_of_week, include_weekend=include_weekend)
if __name__ == '__main__':
    app.run(debug=True)
