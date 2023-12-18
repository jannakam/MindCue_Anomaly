import pandas as pd
from serial.tools import list_ports
import serial
import time
import csv
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
import numpy as np
from datetime import datetime
import socketio
from sklearn.ensemble import IsolationForest
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

pd.options.mode.chained_assignment = None

# Setup a SocketIO client
sio = socketio.Client()
sio.connect('http://localhost:9000')

def train_eeg_model(eeg_data):
    eeg_model = IsolationForest(contamination=0.1, random_state=42)
    eeg_model.fit(eeg_data)
    return eeg_model

def train_gsr_model(gsr_data):
    gsr_model = IsolationForest(contamination=0.1, random_state=42)
    gsr_model.fit(gsr_data)
    return gsr_model

# Function to train the model
def train_model(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data[['BPM', 'GSR', 'C3', 'C4', 'F1', 'F2']])
    return model

# function to impute missing BPM values
def impute_missing_values(dataframe, start_row):

    """
    Imputes missing values in the 'BPM' column of the dataframe.
    Imputation is applied only to rows after the specified start_row.
    """
    # Convert BPM to numeric, coercing errors to NaN
    dataframe['BPM'] = pd.to_numeric(dataframe['BPM'], errors='coerce')

    # Apply imputation only to rows after the specified start_row
    if start_row < len(dataframe):
        subset = dataframe['BPM'].iloc[start_row:]

        if subset.notna().any() and (subset >= 60).any():
            subset.loc[subset < 60] = np.nan
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputed_values = imputer.fit_transform(subset.values.reshape(-1, 1)).flatten()

            # Update the original dataframe
            dataframe['BPM'].iloc[start_row:] = imputed_values
    else:
        print("No rows available for imputation after specified start row.")

def apply_filters(data, num_channels, sampling_rate):
    filtered_data = np.copy(data)

    for channel in range(num_channels):  # Iterate over the number of EEG channels
        DataFilter.perform_bandpass(filtered_data[channel], sampling_rate, 13.0, 80.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        # Add other filters here
        # Apply notch (band-stop) filter to remove power line noise at 50 Hz or 60 Hz
        DataFilter.perform_bandstop(filtered_data[channel], sampling_rate, 50.0, 60.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    return filtered_data

def create_isolation_forest(data):
    """
    Creates and trains an Isolation Forest model on the provided data.
    """
    model = IsolationForest(n_estimators=100, contamination=0.1)
    model.fit(data)
    return model

def detect_anomalies_with_ml(model, data, anomaly_threshold=0.5):
    """
    Detects anomalies using the Isolation Forest model.
    Aggregates predictions over the data and returns a single value:
    -1 for anomalous and 1 for non-anomalous.
    'anomaly_threshold' determines the proportion of -1 predictions
    needed to consider the entire window as anomalous.
    """
    predictions = model.predict(data.T)
    anomaly_proportion = np.mean(predictions == -1)
    return -1 if anomaly_proportion > anomaly_threshold else 1


def main():

    # Initialize Arduino Stuff

    # Identify the correct port
    ports = list_ports.comports()
    for port in ports: 
        print(port)

    # Open the serial com
    serialCom = serial.Serial('COM5', 9600)

    # Initialize EEG stuff
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board_descr = BoardShim.get_board_descr(BoardIds.SYNTHETIC_BOARD.value)
    eeg_names = board_descr['eeg_names'].split(',')

    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Start streaming data')

    sampling_rate = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    max_samples = 1

    start_time = time.time()
    selected_eeg_names = ['C3', 'C4', 'F1', 'F2']  # The EEG channels you're interested in
    # Select only the EEG channels you're interested in
    selected_channel_indices = [eeg_names.index(ch) for ch in selected_eeg_names]
    data_buffer = np.array([]).reshape(len(selected_channel_indices), 0)

    # Create a new CSV file
    with open("newfile.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=",")
        selected_eeg_names = ['C3', 'C4', 'F1', 'F2']  # The EEG channels you're interested in
        EEG_names = ','.join([name for name in eeg_names if name in selected_eeg_names])
        sensor_names = ['Timestamp', 'BPM', 'GSR', EEG_names, 'Anomaly_Score', 'Is_Anomaly']

        # Write the header row
        f.write(','.join(sensor_names) + '\n')

    with open("newfile.csv", "a", newline='') as f:  # Use "a" mode for appending
        writer = csv.writer(f, delimiter=",")

        # Initial training with initial data
        initial_data = pd.read_csv('newfile.csv')
        time.sleep(15)

        # Loop through and collect data as it is available
        while True:
            try:
                # Read the line
                s_bytes = serialCom.readline()
                decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')

                # Check if the data line contains two elements (BPM and GSR)
                if ',' in decoded_bytes:

                    # Parse the line
                    bpm, gsr = decoded_bytes.split(',')


                # Get the current date and time
                current_dnt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_time = time.time()
                data = board.get_current_board_data(max_samples)

                if data.size > 0:
                    eeg_data = data[eeg_channels, :]
                    selected_eeg_data = eeg_data[selected_channel_indices, :]

                    filtered_eeg_data = apply_filters(selected_eeg_data, len(selected_channel_indices), sampling_rate)

                    # Append the filtered data to the data buffer
                    if data_buffer.shape[1] == 0:  # If data_buffer is empty
                        data_buffer = np.empty((len(selected_channel_indices), 0))

                    data_buffer = np.append(data_buffer, filtered_eeg_data, axis=1)

                    # Keep only the latest max_samples data points in the data buffer
                    if data_buffer.shape[1] > max_samples:
                        data_buffer = data_buffer[:, -max_samples:]

                    if current_time - start_time >= 15:
                        # Flatten the data buffer for CSV writing
                        flattened_eeg_data = data_buffer.flatten().tolist()
                        data_buffer = np.empty((len(selected_channel_indices), 0))  # Reset data_buffer
                        start_time = time.time()


                # Create the row
                row = [current_dnt, bpm, gsr] + flattened_eeg_data

                # Write to CSV
                writer.writerow(row)
                f.flush()

                # Read the updated data
                updated_data = pd.read_csv('newfile.csv')

                imputed_bpm = impute_missing_values(updated_data, 10)

                # updated_data['BPM'] = imputed_bpm['BPM']
                updated_data.to_csv('newfile.csv', index=False)

                # Retrain the model on non-anomalous rows if more than 5 rows
                if len(updated_data) > 10:
                    # Split the data so that anomalies are only removed after 30 rows
                    first_10_rows = updated_data.iloc[:10]
                    rows_after_10 = updated_data.iloc[10:]

                    # Filter out anomalous rows from rows after the 30th
                    rows_after_10 = rows_after_10[rows_after_10['Is_Anomaly'] != -1]  # Assuming -1 indicates anomalous

                    # Concatenate the two parts
                    updated_data = pd.concat([first_10_rows, rows_after_10], ignore_index=True)

                    eeg_model = train_eeg_model(updated_data[['C3', 'C4', 'F1', 'F2']])
                    gsr_model = train_gsr_model(updated_data[['BPM','GSR']])
                else:
                    eeg_model = train_eeg_model(updated_data[['C3', 'C4', 'F1', 'F2']])
                    gsr_model = train_gsr_model(updated_data[['BPM','GSR']])

                # model_IF = train_model(updated_data[['BPM', 'GSR']])

                # Process only the latest row
                latest_row = updated_data.iloc[-1:]
                eeg_anomaly = eeg_model.predict(latest_row[['C3', 'C4', 'F1', 'F2']])
                gsr_anomaly = gsr_model.predict(latest_row[['BPM','GSR']])

                eeg_score = eeg_model.decision_function(latest_row[['C3', 'C4', 'F1', 'F2']])
                gsr_score = gsr_model.decision_function(latest_row[['BPM','GSR']])

                anomaly_score = (eeg_score + gsr_score) / 2
                is_anomaly = -1 if (eeg_anomaly == -1 and gsr_anomaly == -1) else 1

                # # Update the latest row with anomaly information
                updated_data.at[updated_data.index[-1], 'Anomaly_Score'] = anomaly_score
                updated_data.at[updated_data.index[-1], 'Is_Anomaly'] = is_anomaly
                
                updated_data.to_csv('newfile.csv', index=False)
                f.flush()


                # Send the data to Flask server
                try:
                    sio.emit('anomaly_data', is_anomaly)
                                            
                except Exception as e:
                    print("Error sending data to Flask server:", e)


                # Print the anomaly information for the latest row
                print(f"Latest Data: {latest_row}")

            except Exception as e:
                print(e)


if __name__ == "__main__":
    main()