import pandas as pd
from serial.tools import list_ports
import serial
import time
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import IterativeImputer
import numpy as np
from datetime import datetime
from sendData import *

# Identify the correct port
ports = list_ports.comports()
for port in ports: 
    print(port)

# Open the serial com
serialCom = serial.Serial('COM5', 9600)

# Toggle DTR to reset the Arduino
# serialCom.setDTR(False)
# time.sleep(1)
# serialCom.flushInput()
# serialCom.setDTR(True)

# Function to train the model
def train_model(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data[['BPM', 'GSR']])
    return model

def impute_missing_values(data):
    imputer = IterativeImputer(random_state=42)
    return pd.DataFrame(imputer.fit_transform(data[['BPM', 'GSR']]), columns=data.columns)

with open("try.csv", "a", newline='') as f:  # Use "a" mode for appending
    writer = csv.writer(f, delimiter=",")

    # Initial training with initial data
    initial_data = pd.read_csv('try.csv')

    # Loop through and collect data as it is available
    while True:
        try:
            # Read the line
            s_bytes = serialCom.readline()
            decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')

            # Check if the data line contains two elements (BPM and GSR)
            if ',' in decoded_bytes:
                # Get the current date and time
                current_dnt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Parse the line
                bpm, gsr = decoded_bytes.split(',')

                # Create the row
                row = [current_dnt, bpm, gsr]
                # print(row)

                # Write to CSV
                writer.writerow(row)
                f.flush()

                # Read the updated data
                updated_data = pd.read_csv('try.csv')
                model_Impute = impute_missing_values(updated_data)
                model_IF = train_model(model_Impute)

                # Process only the latest row
                latest_row = updated_data.iloc[-1:]

                anomaly_score = model_IF.decision_function(latest_row[['BPM', 'GSR']])
                is_anomaly = model_IF.predict(latest_row[['BPM', 'GSR']])

                # Update the latest anomaly result
                latest_anomaly_result["anomaly_score"] = anomaly_score.tolist()  # Convert numpy array to list
                latest_anomaly_result["is_anomaly"] = is_anomaly.tolist()

                # Print the anomaly information for the latest row
                print(f"Latest Data: {latest_row}")
                print(f"Anomaly Score: {anomaly_score}, Is Anomaly: {is_anomaly}")

                # Wait for some time before checking for new data
                time.sleep(1)  # Adjust the sleep time as needed

        except Exception as e:
            print(e)


# # Loop for real-time processing
# while True:
#     try:
#         # Read the updated data
#         updated_data = pd.read_csv('try.csv')

#         # Process only the latest row
#         latest_row = updated_data.iloc[-1:]

#         if 'BPM' in latest_row.columns and 'GSR' in latest_row.columns:
#             anomaly_score = model_IF.decision_function(latest_row[['BPM', 'GSR']])
#             is_anomaly = model_IF.predict(latest_row[['BPM', 'GSR']])

#             # Print the anomaly information for the latest row
#             print(f"Latest Data: {latest_row}")
#             print(f"Anomaly Score: {anomaly_score}, Is Anomaly: {is_anomaly}")

#         # Wait for some time before checking for new data
#         time.sleep(1)  # Adjust the sleep time as needed

#     except Exception as e:
#         print(f"An error occurred: {e}")

# Plotting the data
# def outlier_plot(data, outlier_method_name, x_var, y_var, xaxis_limits=[0,1], yaxis_limits=[0,1]):
#     print(f'Outlier Method: {outlier_method_name}')
#     method = f'{outlier_method_name}_anomaly'
#     print(f'Number of anomalous values: {len(data[data[anomaly] == -1])}')
#     print(f'Total Number of Values: {len(data)}')

#     g = sns.FaceGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[-1,1])
#     g.map(sns.scatterplot, x_var, y_var)
#     g.set(xlim=xaxis_limits, ylim=yaxis_limits)
#     axes = g.axes.flatten()
#     axes[0].set_title(f'Outliers{len(data[data['anomaly'] == -1])} points')
#     axes[1].set_title(f'Inliers{len(data[data['anomaly'] == 1])} points')
#     return g


# outlier_plot(df, 'Isolation Forest', 'Heart Rate', 'SC', [0, 200], [0, 200])