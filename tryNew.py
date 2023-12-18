import pandas as pd
from serial.tools import list_ports
import serial
import time
from datetime import datetime
from sklearn.ensemble import IsolationForest
import numpy as np
import csv
from sklearn.impute import SimpleImputer

pd.options.mode.chained_assignment = None

# Identify the correct port
ports = list_ports.comports()
for port in ports: 
    print(port)

# Open the serial com
serialCom = serial.Serial('COM5', 9600)

# Function to train the model
def train_model(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(data[['BPM', 'GSR']])
    return model

# function to impute missing BPM values
def impute_missing_values(dataframe):
    dataframe['BPM'] = pd.to_numeric(dataframe['BPM'], errors='coerce')
    dataframe.loc[dataframe['BPM'] < 60, 'BPM'] = np.nan
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    dataframe['BPM'] = imputer.fit_transform(dataframe[['BPM']]).flatten()
    return dataframe

# Setup CSV file
with open("newfile.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(['Timestamp', 'BPM', 'GSR', 'Anomaly_Score', 'Is_Anomaly'])

def main():
    last_time = time.time()
    data_buffer = []
    model_IF = None

    while True:
        current_time = time.time()

        if current_time - last_time >= 2:
            if data_buffer:
                df_buffer = pd.DataFrame(data_buffer, columns=['Timestamp', 'BPM', 'GSR'])

                # Impute missing values
                df_buffer = impute_missing_values(df_buffer)

                if model_IF is None:
                    model_IF = train_model(df_buffer[['BPM', 'GSR']])
                else:
                    anomaly_score = model_IF.decision_function(df_buffer[['BPM', 'GSR']])
                    is_anomaly = model_IF.predict(df_buffer[['BPM', 'GSR']])
                    df_buffer['Anomaly_Score'] = anomaly_score
                    df_buffer['Is_Anomaly'] = is_anomaly

                # Write the processed data to CSV
                df_buffer.to_csv('newfile.csv', mode='a', header=False, index=False)

                data_buffer = []
                last_time = current_time

        # Read and accumulate new data
        try:
            s_bytes = serialCom.readline()
            decoded_bytes = s_bytes.decode("utf-8").strip('\r\n')
            if ',' in decoded_bytes:
                bpm, gsr = decoded_bytes.split(',')
                current_dnt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                data_buffer.append([current_dnt, bpm, gsr])
        except Exception as e:
            print(e)

        time.sleep(0.1)

if __name__ == "__main__":
    main()