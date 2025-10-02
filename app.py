from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

app = Flask(__name__)

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
        industry_sales = data['Sales'].sum()  # Total sales
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

if __name__ == '__main__':
    app.run(debug=True)
