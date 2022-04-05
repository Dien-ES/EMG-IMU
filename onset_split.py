import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
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


def set_onset_dataset(group):
    with open(f'../parameter/onset/onset_info_{group}.json', 'r') as f:
        onset_info = json.load(f)

    onset_dataset = {}
    for sid in tqdm(onset_info[group].keys()):
        subject = Subject(group, sid)
        onset_subjects = {}
        for day in onset_info[group][sid].keys():
            if day != 'n_onset':
                if day not in onset_subjects.keys():
                    onset_subjects[day] = {}
                    bbs = subject.days[day].BBS
                    onset_subjects[day]['BBS'] = int(bbs)
                for motion in onset_info[group][sid][day].keys():
                    if motion not in onset_subjects[day].keys():
                        onset_subjects[day][motion] = []

                    emg = subject.days[day].indivs[motion][0].emg.raw
                    for start, end in onset_info[group][sid][day][motion]:
                        bool_idx = (emg.xrange > start) & (emg.xrange < end)
                        data = EMG(emg.data[bool_idx, :], preprocessing=True)
                        # onset_subjects[day][motion] += [data]
                        s, m, _ = motion.split('_')
                        col = COLILOC[s][m]
                        onset_subjects[day][motion] += [
                            np.nanmax(data.rms.data, axis=0)[col]]

        onset_dataset[sid] = onset_subjects

    with open(f'../parameter/onset/{group}_onset_dataset.json', 'w') as f:
        json.dump(onset_dataset, f)


def set_onset_parameter(group):
    with open(f'../parameter/onset/{group}_onset_dataset.json', 'r') as f:
        onset_dataset = json.load(f)

    parameter = {}
    for sid in onset_dataset.keys():
        if sid not in parameter.keys():
            parameter[sid] = {}
        for day in onset_dataset[sid].keys():
            if day not in parameter[sid].keys():
                parameter[sid][day] = {}
                parameter[sid][day]['BBS'] = onset_dataset[sid][day]['BBS']

            for sensor, motion in [('TFL', 'habd'), ('QF', 'ke'), ('GC', 'apf'),
                                   ('TA', 'adf')]:
                if sensor not in parameter[sid][day].keys():
                    parameter[sid][day][sensor] = {}

                try:
                    rms_max = np.array(
                        onset_dataset[sid][day][f'r_{motion}_max'])
                except:
                    rms_max = None

                try:
                    mvc = rms_max / np.mean(
                        onset_dataset[sid][day][f'l_{motion}_max'])
                except:
                    mvc = None

                try:
                    rms_min = np.array(
                        onset_dataset[sid][day][f'r_{motion}_min'])
                except:
                    rms_min = None

                try:
                    submvc = rms_min / np.mean(
                        onset_dataset[sid][day][f'l_{motion}_min'])
                except:
                    submvc = None

                parameter[sid][day][sensor]['RMS_max'] = rms_max
                parameter[sid][day][sensor]['RMS_min'] = rms_min
                parameter[sid][day][sensor]['MVC'] = mvc
                parameter[sid][day][sensor]['subMVC'] = submvc

    with open(f'../parameter/onset/{group}_onset_parameter.pkl', 'wb') as f:
        pickle.dump(parameter, f)
