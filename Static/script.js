document.addEventListener("DOMContentLoaded", function() {
    // Fetch data from Flask backend
    fetch('/get-sales-forecast')
        .then(response => {
            // Check if response is successful
            if (!response.ok) {
                throw new Error('Error fetching data from the backend');
            }
            return response.json();
        })
        .then(data => {
            // Check if the necessary data is available in the response
            if (!data.months || !data.sales || !data.bestSellingProducts || !data.recommendations) {
                throw new Error('Missing data from the backend');
            }

            // Create the sales forecast chart
            const ctx = document.getElementById('salesForecastChart').getContext('2d');
            const salesForecastChart = new Chart(ctx, {
                type: 'line', // Line chart for forecasted sales
                data: {
                    labels: data.months, // Months for the x-axis
                    datasets: [{
                        label: 'Predicted Sales',
                        data: data.sales, // Forecasted sales data
                        borderColor: 'rgba(75, 192, 192, 1)', // Line color
                        fill: false, // Don't fill the area under the line
                        tension: 0.1 // Makes the line smooth
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true // Start the y-axis from zero
                        }
                    }
                }
            });

            // Populate the top-selling products table
            let tableContent = '';
            if (data.bestSellingProducts && data.bestSellingProducts.length > 0) {
                data.bestSellingProducts.forEach(product => {
                    tableContent += `<tr><td>${product.name}</td><td>${product.forecastedSales}</td></tr>`;
                });
                document.getElementById('bestSellingProducts').innerHTML = tableContent;
            } else {
                document.getElementById('bestSellingProducts').innerHTML = '<tr><td colspan="2">No data available</td></tr>';
            }

            // Populate the actionable recommendations list
            let recommendations = '';
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(recommendation => {
                    recommendations += `<li>${recommendation}</li>`;
                });
                document.getElementById('recommendationsList').innerHTML = recommendations;
            } else {
                document.getElementById('recommendationsList').innerHTML = '<li>No recommendations available</li>';
            }

        })
        .catch(error => {
            console.error('Error fetching forecast data:', error);

            // Handle errors by updating the UI with an error message
            document.getElementById('industrySales').innerText = "Error";
            document.getElementById('forecastAccuracy').innerText = "Error";
            document.getElementById('activeAlerts').innerText = "Error";
            document.getElementById('inventoryTurnover').innerText = "Error";
            document.getElementById('salesForecastChart').innerHTML = '<p>Error loading forecast data</p>';
            document.getElementById('bestSellingProducts').innerHTML = '<tr><td colspan="2">Error loading best-selling products</td></tr>';
            document.getElementById('recommendationsList').innerHTML = '<li>Error loading recommendations</li>';
        });
});
document.addEventListener("DOMContentLoaded", function() {
    // Fetch data from Flask backend
    fetch('/get-store-sales-forecast')
        .then(response => {
            if (!response.ok) {
                throw new Error('Error fetching data from the backend');
            }
            return response.json();
        })
        .then(data => {
            // Create the product demand forecast chart
            const ctx = document.getElementById('storeForecastChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.months,
                    datasets: [{
                        label: 'Forecasted Sales',
                        data: data.sales,
                        borderColor: 'rgba(75, 192, 192, 1)',
                        fill: false,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true // Start the y-axis from zero
                        }
                    }
                }
            });

            // Populate the restocking recommendations list
            let recommendations = '';
            data.recommendations.forEach(rec => {
                recommendations += `<li>${rec}</li>`;
            });
            document.getElementById('restockingList').innerHTML = recommendations;
        })
        .catch(error => {
            console.error('Error fetching data:', error);
            // Handle errors by updating the UI with an error message
            document.getElementById('restockingList').innerHTML = '<li>Error loading restocking recommendations</li>';
            document.getElementById('storeForecastChart').innerHTML = '<p>Error loading forecast data</p>';
        });
});
