import time
import numpy as np
import pandas as pd
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter

def main():
    BoardShim.enable_dev_board_logger()

    # Use synthetic board for demo
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'Start streaming data')

    try:
        while True:
            # Fetching data from the board
            data = board.get_current_board_data(256)  # Get last 256 samples
            if data.size > 0:
                eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
                df = pd.DataFrame(np.transpose(data))
                print(df.head(10))  # Print the latest data

                # Append data to CSV file
                DataFilter.write_file(data, 'test.csv', 'w')

            time.sleep(1)  # Small delay to prevent overwhelming your terminal
    except KeyboardInterrupt:
        # Gracefully stop the stream and release session when user interrupts
        print("Stream stopped by the user.")
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()