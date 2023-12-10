import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes

def apply_filters(data, eeg_channels, sampling_rate):
    filtered_data = np.copy(data)

    for channel in eeg_channels:
        DataFilter.perform_bandpass(filtered_data[channel], sampling_rate, 15.0, 30.0, 4, FilterTypes.BUTTERWORTH.value, 0)
        # Add other filters here

    return filtered_data

def plot_real_time(fig, ax, data, max_samples):
    ax.clear()
    ax.plot(data)
    ax.set_xlim(0, max_samples)
    ax.set_ylim(-100, 100)  # Set Y axis limits appropriate for your data
    plt.pause(0.001)

def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Start streaming data')

    sampling_rate = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    max_samples = 20

    # Set up real-time plot
    plt.ion()
    fig, ax = plt.subplots()
    plt.show()

    try:
        while True:
            data = board.get_current_board_data(max_samples)
            if data.size > 0:
                filtered_data = apply_filters(data, eeg_channels, sampling_rate)

                # Transpose data for plotting
                transposed_data = np.transpose(filtered_data)
                plot_real_time(fig, ax, transposed_data, max_samples)

                # Append data to CSV file
                DataFilter.write_file(filtered_data, 'test.csv', 'a')

            time.sleep(1)
    except KeyboardInterrupt:
        print("Stream stopped by the user.")
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()