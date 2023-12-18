import pandas as pd
from serial.tools import list_ports
import serial
import time
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import numpy as np
from datetime import datetime
import socketio
import tkinter as tk
import pandas as pd

pd.options.mode.chained_assignment = None

# Identify the correct port
ports = list_ports.comports()
for port in ports: 
    print(port)

# Open the serial com
serialCom = serial.Serial('COM5', 9600)

# Setup a SocketIO client
sio = socketio.Client()
sio.connect('http://localhost:9000')

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

# function to impute missing BPM values
def impute_missing_values(dataframe, start_row):

    dataframe['BPM'] = pd.to_numeric(dataframe['BPM'], errors='coerce')

    # Apply imputation only to rows after the specified start_row
    if start_row < len(dataframe):
        subset = dataframe.iloc[start_row:]
        
        if subset['BPM'].notna().any() and (subset['BPM'] >= 60).any():
            subset.loc[subset['BPM'] < 60, 'BPM'] = np.nan
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            subset['BPM'] = imputer.fit_transform(subset[['BPM']]).flatten()
            
            # Update the original dataframe
            dataframe.iloc[start_row:] = subset
    else:
        print("No rows available for imputation after specified start row.")

with open("newfile.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=",")
    # Write the header row
    writer.writerow(['Timestamp', 'BPM', 'GSR', 'Anomaly_Score', 'Is_Anomaly'])


with open("newfile.csv", "a", newline='') as f:  # Use "a" mode for appending
    writer = csv.writer(f, delimiter=",")

    # Initial training with initial data
    initial_data = pd.read_csv('newfile.csv')

    last_time = time.time()
    data_buffer = []
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
                updated_data = pd.read_csv('newfile.csv')
                                
                # To compare affect of imputing
                # print(f'Data before imputing {latest_row}')

                imputed_bpm = impute_missing_values(updated_data, 10)
                # updated_data['BPM'] = imputed_bpm['BPM']
                updated_data.to_csv('newfile.csv', index=False)
                last_row_number = len(updated_data) - 1

                # Retrain the model on non-anomalous rows if more than 5 rows
                if len(updated_data) > 10:
                    # Split the data so that anomalies are only removed after 30 rows
                    first_30_rows = updated_data.iloc[:10]
                    rows_after_30 = updated_data.iloc[10:]

                    # Filter out anomalous rows from rows after the 30th
                    rows_after_30 = rows_after_30[rows_after_30['Is_Anomaly'] != -1]  # Assuming -1 indicates anomalous

                    # Concatenate the two parts
                    updated_data = pd.concat([first_30_rows, rows_after_30], ignore_index=True)

                    model_IF = train_model(updated_data[['BPM', 'GSR']])
                else:
                    model_IF = train_model(updated_data[['BPM', 'GSR']])

                # model_IF = train_model(updated_data[['BPM', 'GSR']])

                # Process only the latest row
                latest_row = updated_data.iloc[-1:]
                anomaly_score = model_IF.decision_function(latest_row[['BPM', 'GSR']])
                is_anomaly = model_IF.predict(latest_row[['BPM', 'GSR']])

                # # Update the latest row with anomaly information
                updated_data.at[updated_data.index[-1], 'Anomaly_Score'] = anomaly_score
                updated_data.at[updated_data.index[-1], 'Is_Anomaly'] = is_anomaly
                
                updated_data.to_csv('newfile.csv', index=False)
                f.flush()

                # Send the data to Flask server
                try:
                    sio.emit('anomaly_data', is_anomaly.tolist())
                                            
                except Exception as e:
                    print("Error sending data to Flask server:", e)


                # Print the anomaly information for the latest row
                print(f"Latest Data: {latest_row}")

                # Wait for some time before checking for new data
                time.sleep(0.1)  # Adjust the sleep time as needed

        except Exception as e:
            print(e)

