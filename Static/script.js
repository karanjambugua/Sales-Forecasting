document.addEventListener("DOMContentLoaded", function() {
    // Fetch data for the sales forecast dashboard
    fetch('/get-sales-forecast')
        .then(response => {
            if (!response.ok) {
                throw new Error('Error fetching data from the backend');
            }
            return response.json();
        })
        .then(data => {
            console.log(data);  // Debugging the response

            // Check if the necessary data is available in the response
            if (!data.industrySales || !data.forecastAccuracy || !data.activeAlerts || !data.inventoryTurnover) {
                throw new Error('Missing data from the backend');
            }

            // Populate the totals for the dashboard
            document.getElementById('industrySales').innerText = data.industrySales || "Data not available";
            document.getElementById('forecastAccuracy').innerText = data.forecastAccuracy || "Data not available";
            document.getElementById('activeAlerts').innerText = data.activeAlerts || "Data not available";
            document.getElementById('inventoryTurnover').innerText = data.inventoryTurnover || "Data not available";

        })
        .catch(error => {
            console.error('Error fetching forecast data:', error);

            // Handle errors by updating the UI with an error message
            document.getElementById('industrySales').innerText = "Error";
            document.getElementById('forecastAccuracy').innerText = "Error";
            document.getElementById('activeAlerts').innerText = "Error";
            document.getElementById('inventoryTurnover').innerText = "Error";
        });
});

// Fetch data for the store-specific sales forecast
document.addEventListener("DOMContentLoaded", function() {
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
            if (data.recommendations && data.recommendations.length > 0) {
                data.recommendations.forEach(rec => {
                    recommendations += `<li>${rec}</li>`;
                });
                document.getElementById('restockingList').innerHTML = recommendations;
            } else {
                document.getElementById('restockingList').innerHTML = '<li>No restocking recommendations available</li>';
            }
        })
        .catch(error => {
            console.error('Error fetching store sales data:', error);
            // Handle errors by updating the UI with an error message
            document.getElementById('restockingList').innerHTML = '<li>Error loading restocking recommendations</li>';
            document.getElementById('storeForecastChart').innerHTML = '<p>Error loading forecast data</p>';
        });
});
document.addEventListener("DOMContentLoaded", function() {
    const ctx = document.getElementById('salesForecastChart');
    if (ctx) {
        const salesForecastChart = new Chart(ctx, {
            // chart configuration
        });
    }
});
// JavaScript to toggle the sidebar visibility
document.getElementById('hamburger').addEventListener('click', function() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('active');  // Toggle the 'active' class
});
