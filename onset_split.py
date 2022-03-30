import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
sns.set_style("whitegrid")

from data_load import *

DATA_PATH = '../../../../Database/EMG_IMU'
COLILOC = {'r':{'habd':2, 'ke':3, 'apf':4, 'adf':5},
           'l':{'habd':8, 'ke':9, 'apf':10, 'adf':11}}


def specific_sensor_plot(data, set_onset=None):
    signal = data.emg.raw
    side, motion, power = data.motion.split('_')
    col = COLILOC[side][motion]
    _, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.set_title(data.motion, fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_ylabel('Amplitude [mV]', fontsize=20)
    ax.set_xlabel('Time [s]', fontsize=20)
    sns.lineplot(ax=ax,
                 x=signal.xrange[:signal.data.shape[0]],
                 y=signal.data[:, col])
    plt.xticks(range(int(signal.xrange[-1] + 2)))
    plt.show()

    if set_onset:
        trial = int(input())
        onset = []
        for _ in range(trial):
            start, end = map(int, input().split())
            onset.append((start, end))

        return onset


def set_onset_info():
    try:
        with open('../onset_info.pkl', 'rb') as f:
            onset_info = pickle.load(f)
    except:
        onset_info = {'OnsetID': [], 'Group': [], 'Sid': [], 'Day': [], 'Motion': [],
                      'Onset': []}
        with open('../onset_info.pkl', 'wb') as f:
            pickle.dump(onset_info, f)

    onset_id = -1
    for group in ['Healthy', 'Disabed']:
        if group == 'Healthy':
            end_sid = 20
            sid_list = range(1, end_sid + 1)
        else:
            end_sid = 22
            sid_list = list(range(1, end_sid + 1))
            sid_list.remove(13)
            sid_list.remove(17)

        for sid in sid_list:
            print(f'---------------------{group} {sid}---------------------')
            subject = Subject(group, sid)
            print(f'subject load complete.')

            for days in subject.days:
                for signals in days.indivs:
                    onset_id += 1
                    if onset_id not in onset_info['OnsetID']:
                        data = signals[0]
                        onset = specific_sensor_plot(data, set_onset=True)

                        onset_info['OnsetID'] += [onset_id]
                        onset_info['Group'] += [data.group]
                        onset_info['Sid'] += [data.sid]
                        onset_info['Day'] += [data.day]
                        onset_info['Motion'] += [data.motion]
                        onset_info['Onset'] += [onset]

                        with open('../onset_info.pkl', 'wb') as f:
                            pickle.dump(onset_info, f)
