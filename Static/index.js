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
            if (!data.months || !data.sales) {
                throw new Error('Missing data from the backend');
            }

            // Populate other metrics like Industry Sales, Forecast Accuracy, etc.
             document.getElementById('industrySales').innerText = ` ${data.industrySales.toLocaleString()}` || "Data not available";
            document.getElementById('forecastAccuracy').innerText = data.forecastAccuracy || "Data not available";
            document.getElementById('activeAlerts').innerText = data.activeAlerts || "Data not available";
            document.getElementById('inventoryTurnover').innerText = data.inventoryTurnover || "Data not available";

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
            document.getElementById('recommendationsList').innerHTML = '<li>Error loading recommendations</li>';
        });
});
// JavaScript to toggle the sidebar visibility
document.getElementById('hamburger').addEventListener('click', function() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('active');  // Toggle the 'active' class
});
