import pandas as pd
from serial.tools import list_ports
import serial
import time
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import numpy as np

while True:
    df = pd.read_csv('data.csv')

    # Choose the columns we want to apply the detection on
    anomaly_inputs = ['Heart Rate', 'SC']

    # The model
    model_IF = IsolationForest(contamination=0.1, random_state=42)
    model_IF.fit(df[anomaly_inputs])

    # Defining what the model will be looking at (our columns)
    df['anomaly_scores'] = model_IF.decision_function(df[anomaly_inputs])
    df['anomaly'] = model_IF.predict(df[anomaly_inputs])
    df.loc[:, ['Heart Rate', 'SC', 'anomaly_scores', 'anomaly']]
    print(df.head())

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