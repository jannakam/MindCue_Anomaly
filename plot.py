import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import threading

# Function to load data
def load_data():
    df = pd.read_csv('newfile.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df

# Initial load of the DataFrame
df = load_data()
print("DataFrame loaded successfully. Number of rows:", len(df))

# Initialize the plot with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Readings for the first and second subplot
readings1 = ['BPM', 'GSR']
readings2 = ['C3', 'C4', 'F1', 'F2']
colors1 = ['paleturquoise', 'darkseagreen']
colors2 = ['thistle', 'powderblue', 'salmon', 'khaki']

new_data_loaded = False

# Initialize legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

def update(frame):
    global df, new_data_loaded
    if new_data_loaded:
        ax1.clear()
        ax2.clear()

    # Plot BPM and GSR on the first subplot
    for reading, color in zip(readings1, colors1):
        ax1.plot(df['Timestamp'][:frame], df[reading][:frame], label=reading, color=color)
        # Plot anomalies for BPM and GSR
        anomalies = df[(df['Is_Anomaly_BPMGSR'] == -1) & (df.index <= frame)]
        ax1.scatter(anomalies['Timestamp'], anomalies[reading], color='darkred', edgecolor='black', marker='o')

    # Plot EEG readings on the second subplot
    for reading, color in zip(readings2, colors2):
        ax2.plot(df['Timestamp'][:frame], df[reading][:frame], label=reading, color=color)
        # Plot anomalies for EEG readings
        anomalies = df[(df['Is_Anomaly_EEG'] == -1) & (df.index <= frame)]
        ax2.scatter(anomalies['Timestamp'], anomalies[reading], color='blue', edgecolor='black', marker='o')

    ## Recreate the legends after clearing
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    new_data_loaded = True

    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper left')
    ax1.set_ylabel('BPM & GSR')
    ax2.set_ylabel('EEG Readings')
    ax2.set_xlabel('Timestamp')
    # new_data_loaded = False

# Legends are created once here
for reading, color in zip(readings1, colors1):
    ax1.plot([], [], label=reading, color=color)
for reading, color in zip(readings2, colors2):
    ax2.plot([], [], label=reading, color=color)
ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

# Create animation
ani = FuncAnimation(fig, update, interval=500, repeat=False)
plt.tight_layout()

# Function to refresh data
def refresh_data():
    global df, new_data_loaded
    while True:
        try:
            new_df = load_data()
            if len(new_df) != len(df):
                df = new_df
                new_data_loaded = True
        except Exception as e:
            print("Error reading file:", e)
        time.sleep(5)  # Wait for 5 seconds before refreshing again

# Run the refresh_data function in a separate thread
thread = threading.Thread(target=refresh_data)
thread.start()

plt.show()