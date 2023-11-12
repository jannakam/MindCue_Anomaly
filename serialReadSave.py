from serial.tools import list_ports
import serial
import time
import csv
from datetime import datetime

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

# How many data points to record
kmax = 180*90

with open("try.csv", "a", newline='') as f:  # Use "a" mode for appending
    writer = csv.writer(f, delimiter=",")

    # Loop through and collect data as it is available
    for k in range(kmax):
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
                print(row)

                # Write to CSV
                writer.writerow(row)
                f.flush()

        except Exception as e:
            print(e)
