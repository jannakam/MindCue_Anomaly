function fetchAnomalyResults() {
    // URL of the Flask server endpoint
    const url = 'http://localhost:5000/get_anomaly_result';

    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Anomaly Detection Results:', data);

            // Process the data as needed
            // For example, update the UI with this data
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
        });
}

// Call the function to fetch results
fetchAnomalyResults();

// Optionally, set an interval to fetch results periodically
setInterval(fetchAnomalyResults, 5000); //Fetch results every 5000 milliseconds (5 seconds)