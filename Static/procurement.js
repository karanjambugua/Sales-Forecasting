document.addEventListener("DOMContentLoaded", function () {
    // Fetch procurement data from the Flask backend
    fetch('/get-procurement-data')
        .then(response => response.json())
        .then(data => {
            // Populate the product demand forecast chart (Line and Pie charts for better representation)
            const ctx = document.getElementById('procurementChart').getContext('2d');
            new Chart(ctx, {
                type: 'line', // Change to 'line' for trend forecast
                data: {
                    labels: data.categories,
                    datasets: [
                        {
                            label: 'Stock vs Demand',
                            data: data.stockAndDemand,
                            backgroundColor: 'rgba(54, 162, 235, 0.2)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1,
                            fill: false,
                            tension: 0.1
                        },
                        {
                            label: 'Replenishment Need',
                            data: data.replenishmentNeed,
                            backgroundColor: 'rgba(255, 159, 64, 0.2)',
                            borderColor: 'rgba(255, 159, 64, 1)',
                            borderWidth: 1,
                            fill: false,
                            tension: 0.1
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            enabled: true
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Populate the replenishment suggestions grid
            let replenishment = '';
            data.replenishmentSuggestions.forEach(item => {
                replenishment += `<div class="recommendation" data-reason="${item.reason}">${item.product}</div>`;
            });
            document.getElementById('replenishmentList').innerHTML = replenishment;

            // Handle recommendation click events
            document.querySelectorAll('.recommendation').forEach(item => {
                item.addEventListener('click', function () {
                    alert(item.dataset.reason);
                });
            });

            // Populate the top-performing products
            let topProducts = '';
            data.topPerformingProducts.forEach(item => {
                topProducts += `<li>${item.product}: Ksh ${item.sales}</li>`;
            });
            document.getElementById('topPerformingProducts').innerHTML = topProducts;

            // Populate the underperforming products
            let underProducts = '';
            data.underperformingProducts.forEach(item => {
                underProducts += `<li>${item.product}: Ksh ${item.sales}</li>`;
            });
            document.getElementById('underperformingProducts').innerHTML = underProducts;

            // Populate Supplier Performance (mock data, replace with actual data)
            let supplierPerformance = `
                <p>Supplier A: 95% Order Fulfillment</p>
                <p>Supplier B: 90% Order Fulfillment</p>
                <p>Supplier C: 88% Order Fulfillment</p>
            `;
            document.getElementById('supplierPerformance').innerHTML = supplierPerformance;

            // Alerts (display alert categories)
            let procurementAlerts = '';
            data.procurementAlerts.forEach(alert => {
                procurementAlerts += `<li>${alert}</li>`;
            });
            document.getElementById('procurementAlerts').innerHTML = procurementAlerts;

        })
        .catch(error => {
            console.error('Error fetching data:', error);
            // Display error message if data fetching fails
            document.getElementById('replenishmentList').innerHTML = '<li>Error loading replenishment suggestions</li>';
            document.getElementById('procurementChart').innerHTML = '<p>Error loading chart data</p>';
        });
});
// JavaScript to toggle the sidebar visibility
document.getElementById('hamburger').addEventListener('click', function() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('active');  // Toggle the 'active' class
});
