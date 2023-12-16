import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

def apply_filters(data, num_channels, sampling_rate):
    filtered_data = np.copy(data)

    for channel in range(num_channels):  # Iterate over the number of EEG channels
        DataFilter.perform_bandpass(filtered_data[channel], sampling_rate, 13.0, 80.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        # Add other filters here
        # Apply notch (band-stop) filter to remove power line noise at 50 Hz or 60 Hz
        DataFilter.perform_bandstop(filtered_data[channel], sampling_rate, 50.0, 60.0, 4, FilterTypes.BUTTERWORTH.value, 0)
    return filtered_data

def detect_anomalies(data, threshold=2, anomaly_threshold=0.1):
    """
    Detects anomalies in 2D data array based on Z-score.
    Returns -1 if the percentage of anomalous data exceeds anomaly_threshold, otherwise returns 1.
    """
    z_scores = np.abs((data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True))
    anomalies = np.where(z_scores > threshold, 1, 0)
    anomaly_percentage = np.mean(anomalies, axis=1)
    return -1 if np.any(anomaly_percentage > anomaly_threshold) else 1

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


    try:
        data_buffer = np.array([]).reshape(len(eeg_channels), 0)
        start_time = time.time()
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
                    anomaly_result = detect_anomalies(data_buffer[selected_channels])
                    
                    # Return or store the anomaly result here
                    print(anomaly_result)

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