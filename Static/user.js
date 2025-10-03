document.addEventListener('DOMContentLoaded', () => {
    // Fetch actual data (product names and shops) from your backend or database
    const products = ['Winter Jacket', 'Sneakers', 'Vintage T-shirt']; // Replace with actual products data
    const shops = ['901 - Gikomba', '902 - Downtown']; // Replace with actual shops data

    // Populate product dropdown dynamically
    const productSelect = document.getElementById('product');
    products.forEach(function (product) {
        const option = document.createElement('option');
        option.value = product;
        option.textContent = product;
        productSelect.appendChild(option);
    });

    // Populate shop dropdown dynamically
    const shopSelect = document.getElementById('shop');
    shops.forEach(function (shop) {
        const option = document.createElement('option');
        option.value = shop;
        option.textContent = shop;
        shopSelect.appendChild(option);
    });

    // Show the corresponding form based on user selection
    function showForm(formType) {
        // Hide all forms first
        document.getElementById('productForm').classList.add('d-none');
        document.getElementById('dateForm').classList.add('d-none');
        document.getElementById('shopForm').classList.add('d-none');
        document.getElementById('dayweekForm').classList.add('d-none');

        // Show the selected form
        document.getElementById(formType + 'Form').classList.remove('d-none');
    }

    // Event listeners for form buttons (to show corresponding forms)
    document.getElementById('productButton').addEventListener('click', function () {
        showForm('product');
    });
    document.getElementById('dateButton').addEventListener('click', function () {
        showForm('date');
    });
    document.getElementById('shopButton').addEventListener('click', function () {
        showForm('shop');
    });
    document.getElementById('dayweekButton').addEventListener('click', function () {
        showForm('dayweek');
    });

    // Handle form submission for Product Forecasting
    document.getElementById('productFormDetails').addEventListener('submit', function (e) {
        e.preventDefault();

        const product = document.getElementById('product').value;
        const quantity = document.getElementById('quantity').value;

        // Perform AJAX request to Flask backend for prediction
        fetch('/predict-product', {
            method: 'POST',
            body: new URLSearchParams({
                'product': product,
                'quantity': quantity
            }),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            }
        })
        .then(response => response.json())
        .then(data => {
            // Handle the prediction results
            alert(`Forecast result: ${JSON.stringify(data)}`);
        })
        .catch(error => console.error('Error:', error));
    });

    // Handle form submission for Date-Based Forecasting
    document.getElementById('dateFormDetails').addEventListener('submit', function (e) {
        e.preventDefault();

        const startDate = document.getElementById('startDate').value;
        const endDate = document.getElementById('endDate').value;

        // Perform AJAX request to Flask backend for prediction
        fetch('/predict-date', {
            method: 'POST',
            body: new URLSearchParams({
                'startDate': startDate,
                'endDate': endDate
            }),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            }
        })
        .then(response => response.json())
        .then(data => {
            // Handle the prediction results
            alert(`Forecast result: ${JSON.stringify(data)}`);
        })
        .catch(error => console.error('Error:', error));
    });

    // Handle form submission for Shop Performance Forecasting
    document.getElementById('shopFormDetails').addEventListener('submit', function (e) {
        e.preventDefault();

        const shop = document.getElementById('shop').value;
        const shopSales = document.getElementById('shopSales').value;

        // Perform AJAX request to Flask backend for prediction
        fetch('/predict-shop', {
            method: 'POST',
            body: new URLSearchParams({
                'shop': shop,
                'shopSales': shopSales
            }),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            }
        })
        .then(response => response.json())
        .then(data => {
            // Handle the prediction results
            alert(`Forecast result: ${JSON.stringify(data)}`);
        })
        .catch(error => console.error('Error:', error));
    });

    // Handle form submission for Day/Week/Weekend Forecasting
    document.getElementById('dayweekFormDetails').addEventListener('submit', function (e) {
        e.preventDefault();

        const dayOfWeek = document.getElementById('dayOfWeek').value;
        const includeWeekend = document.getElementById('includeWeekend').checked ? 'on' : 'off';

        // Perform AJAX request to Flask backend for prediction
        fetch('/predict-dayweek', {
            method: 'POST',
            body: new URLSearchParams({
                'dayOfWeek': dayOfWeek,
                'includeWeekend': includeWeekend
            }),
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            }
        })
        .then(response => response.json())
        .then(data => {
            // Handle the prediction results
            alert(`Forecast result: ${JSON.stringify(data)}`);
        })
        .catch(error => console.error('Error:', error));
    });
});
document.addEventListener('DOMContentLoaded', function () {
    // Show the corresponding form based on user selection
    function showForm(formType) {
        // Hide all forms first
        document.getElementById('productForm').classList.add('d-none');
        document.getElementById('dateForm').classList.add('d-none');
        document.getElementById('shopForm').classList.add('d-none');
        document.getElementById('dayweekForm').classList.add('d-none');

        // Show the selected form
        document.getElementById(formType + 'Form').classList.remove('d-none');
    }

    // Event listeners for buttons
    const productBtn = document.getElementById('productBtn');
    const dateBtn = document.getElementById('dateBtn');
    const shopBtn = document.getElementById('shopBtn');
    const dayweekBtn = document.getElementById('dayweekBtn');

    if (productBtn) {
        productBtn.addEventListener('click', function () { showForm('product'); });
    }

    if (dateBtn) {
        dateBtn.addEventListener('click', function () { showForm('date'); });
    }

    if (shopBtn) {
        shopBtn.addEventListener('click', function () { showForm('shop'); });
    }

    if (dayweekBtn) {
        dayweekBtn.addEventListener('click', function () { showForm('dayweek'); });
    }
});
