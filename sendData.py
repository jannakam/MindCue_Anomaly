from flask import Flask, jsonify
import threading

app = Flask(__name__)

# Shared variable to store the latest anomaly detection result
latest_anomaly_result = {"anomaly_score": None, "is_anomaly": None}

# Endpoint to get the latest anomaly detection result
@app.route('/get_anomaly_result', methods=['GET'])
def get_anomaly_result():
    return jsonify(latest_anomaly_result)

# Function to run Flask app
def run_flask_app():
    app.run(host='0.0.0.0', port=5000, debug=True)

# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()