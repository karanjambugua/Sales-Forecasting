document.addEventListener("DOMContentLoaded", function() {
    fetch('/get-store-sales-forecast-v4')  // Fetch from the updated endpoint
        .then(response => response.json())
        .then(data => {
            console.log(data);  // Debugging the response

            // Display Current Sales
            document.getElementById('totalSales').innerText = `Ksh ${data.totalSales}`;
            document.getElementById('avgSales').innerText = `Ksh ${data.avg_sales}`;

            // Create Sales Forecast Chart
            const ctx = document.getElementById('salesForecastChart').getContext('2d');
            const salesForecastChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{
                        label: 'Predicted Sales',
                        data: data.monthly_sales,
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
                            beginAtZero: true
                        }
                    }
                }
            });

            // Populate Top Performing Products
            let topProducts = '';
            for (const [product, sales] of Object.entries(data.top_performing_products)) {
                topProducts += `<li>${product}: Ksh ${sales.toFixed(2)}</li>`;
            }
            document.getElementById('topPerformingProducts').innerHTML = topProducts;

            // Populate Underperforming Products
            let underProducts = '';
            for (const [product, sales] of Object.entries(data.underperforming_products)) {
                underProducts += `<li>${product}: Ksh ${sales.toFixed(2)}</li>`;
            }
            document.getElementById('underperformingProducts').innerHTML = underProducts;

            // Populate Inventory Levels
            let inventory = '';
            for (const [product, qty] of Object.entries(data.inventory_status)) {
                inventory += `<li>${product}: ${qty} units</li>`;
            }
            document.getElementById('inventoryLevels').innerHTML = inventory;

            // Populate Restocking Recommendations
            let recommendations = '';
            data.restock_recommendations.forEach(item => {
                recommendations += `<li><a href="#" class="recommendation" data-reason="${item.reason}">${item.product}</a></li>`;
            });
            document.getElementById('recommendationsList').innerHTML = recommendations;

            // Add click events for recommendations
            document.querySelectorAll('.recommendation').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.preventDefault();
                    item.classList.toggle('open'); // Toggle 'open' class to show reason
                    alert(item.dataset.reason); // Show the reason for the recommendation
                });
            });
            // Customer Feedback
            document.getElementById('customerFeedback').innerHTML = `Rating: ${data.customer_feedback.avg_rating}`;
            let complaints = '';
            data.customer_feedback.common_complaints.forEach(complaint => {
                complaints += `<li>${complaint}</li>`;
            });
            document.getElementById('complaints').innerHTML = complaints;

            // Profit Margin
            document.getElementById('profitMargin').innerHTML = `${(data.profit_margin * 100).toFixed(2)}%`;
        })
        .catch(error => {
            console.error('Error fetching store manager data:', error);
        });
});
// JavaScript to toggle the sidebar visibility
document.getElementById('hamburger').addEventListener('click', function() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('active');  // Toggle the 'active' class
});
