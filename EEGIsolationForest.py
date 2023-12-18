import time
import numpy as np
import pandas as pd
import serial
import socketio
import time
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
from sklearn.ensemble import IsolationForest

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

def detect_anomalies_with_ml(model, data, anomaly_threshold=0.4):
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

    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board_descr = BoardShim.get_board_descr(BoardIds.SYNTHETIC_BOARD.value)
    eeg_names = board_descr['eeg_names'].split(',')
    with open('test.csv', 'w') as f:
        f.write(','.join(eeg_names) + '\n')
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Start streaming data')

    sampling_rate = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    max_samples = 10

    model = None
    data_buffer = np.array([]).reshape(len(eeg_channels), 0)
    start_time = time.time()
    anomaly_results = None

    try:
        while True:
            current_time = time.time()
            data = board.get_current_board_data(max_samples)
            if data.size > 0:
                # Select only the EEG channels
                eeg_data = data[eeg_channels, :]

                # Apply filters to the EEG data
                filtered_eeg_data = apply_filters(eeg_data, len(eeg_channels), sampling_rate)

                # Accumulate data for 2 seconds
                data_buffer = np.append(data_buffer, filtered_eeg_data, axis=1)
                if current_time - start_time >= 2:
                    # Process accumulated data
                    selected_channels = [eeg_names.index(ch) for ch in ['C3', 'C4', 'F1', 'F2']]
                    filtered_data = data_buffer[selected_channels]

                    # Initialize or update the model and detect anomalies
                    if model is None and filtered_data.shape[1] > 0:
                        model = create_isolation_forest(filtered_data.T)
                    elif model is not None:
                        anomaly_result = detect_anomalies_with_ml(model, filtered_data)
                        
                        # Append the anomaly result to each row in the data_buffer
                        anomaly_column = np.full((1, data_buffer.shape[1]), anomaly_result)
                        data_with_anomaly = np.append(data_buffer, anomaly_column, axis=0)

                        # Write the data with the anomaly result to the CSV file
                        DataFilter.write_file(data_with_anomaly, 'test.csv', 'a')
                        
                        # Print or store the anomaly result
                        print(anomaly_result)

                        # Use non-anomalous data to retrain the model
                        if anomaly_result == 1:
                            model = create_isolation_forest(filtered_data.T)

                    # Clear buffer and reset timer
                    data_buffer = np.array([]).reshape(len(eeg_channels), 0)
                    start_time = time.time()

                # Append EEG data to CSV file
                DataFilter.write_file(filtered_eeg_data, 'test.csv', 'a')


            time.sleep(1)

    except KeyboardInterrupt:
        print("Stream stopped by the user.")
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()