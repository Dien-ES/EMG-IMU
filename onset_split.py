import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from data_load import *

sns.set_style("whitegrid")

DATA_PATH = '../../../../Database/EMG_IMU'
COLILOC = {'r': {'habd': 2, 'ke': 3, 'apf': 4, 'adf': 5},
           'l': {'habd': 8, 'ke': 9, 'apf': 10, 'adf': 11}}


def specific_sensor_plot(data, set_onset=None):
    signal = data.emg.raw
    side, motion, power = data.motion.split('_')
    col = COLILOC[side][motion]
    max_time = int(signal.xrange[-1] + 1)
    _, ax = plt.subplots(1, 1, figsize=(5 * max_time // 10, 5))
    ax.set_title(data.motion, fontsize=20)
    ax.tick_params(labelsize=15)
    ax.set_ylabel('Amplitude [mV]', fontsize=20)
    ax.set_xlabel('Time [s]', fontsize=20)
    sns.lineplot(ax=ax,
                 x=signal.xrange[:signal.data.shape[0]],
                 y=signal.data[:, col])
    plt.xticks(range(max_time))
    plt.show()

    if set_onset:
        while True:
            trial = input('----------trial count:')
            if trial == 'exit':
                raise RuntimeError('onset quit.')
            else:
                try:
                    trial = int(trial)
                    break
                except:
                    pass

        onset = []
        cnt = 0
        while cnt < trial:
            start = input(f'onset {cnt} start:')
            end = input(f'onset {cnt} end:')
            if (start == 'exit') or (end == 'exit'):
                raise RuntimeError('onset quit.')
            else:
                try:
                    onset.append((int(start), int(end)))
                    cnt += 1
                except:
                    pass
        return onset


def set_onset_info(group):
    try:
        with open(f'../parameter/onset/onset_info_{group}.json', 'r') as f:
            onset_info = json.load(f)
    except:
        onset_info = {}
        with open(f'../parameter/onset/onset_info_{group}.json', 'w') as f:
            json.dump(onset_info, f)

    if group == 'Healthy':
        end_sid = 20
        sid_list = range(1, end_sid + 1)
    else:
        end_sid = 22
        sid_list = list(range(1, end_sid + 1))
        sid_list.remove(13)
        sid_list.remove(17)
    if group not in onset_info.keys():
        onset_info[group] = {}

    for sid in sid_list:
        print(f'---------------------{group} {sid}---------------------')
        if f'sid_{sid}' not in onset_info[group].keys():
            onset_info[group][f'sid_{sid}'] = {'n_onset': 0}

        path_list = glob.glob(os.path.join(DATA_PATH, group,
                                           f'R{sid:03}_*',
                                           f'indiv_*', '*.csv'))
        if onset_info[group][f'sid_{sid}']['n_onset'] == len(path_list):
            print('already onset exist.')
            continue
        else:
            subject = Subject(group, sid)
            print(f'subject load complete.')

        for day in subject.days:
            for indiv in day.indivs:
                data = indiv.signals[0]
                if f'day_{data.day}' not in onset_info[group][
                    f'sid_{sid}'].keys():
                    onset_info[group][f'sid_{sid}'][f'day_{data.day}'] = {}
                if data.motion not in onset_info[data.group][f'sid_{sid}'][
                    f'day_{data.day}'].keys():
                    onset = specific_sensor_plot(data, set_onset=True)

                    onset_info[group][f'sid_{sid}']['n_onset'] += 1
                    onset_info[data.group][f'sid_{sid}'][f'day_{data.day}'][
                        data.motion] = onset
                    with open(f'../parameter/onset/onset_info_{group}.json', 'w') as f:
                        json.dump(onset_info, f)
