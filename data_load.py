import os
import re
import glob
import pandas as pd

from signal_processing import *

DATA_PATH = '../../../../Database/EMG_IMU'
SENSORS = ['TP', 'LD', 'TFL', 'QF', 'GC', 'TA']
FUNC_MV = ['back_extension', 'back_flexion',
           'sit_to_stand', 'supine_to_sit', 'gait']
INDIV_MV = []
for side in ['r', 'l']:
    for task in ['habd', 'ke', 'apf', 'adf']:
        for power in ['max', 'min']:
            INDIV_MV += [f'{side}_{task}_{power}']


class Subject:
    def __init__(self, group, sid):
        self.group = group
        self.sid = sid
        self.days = []


class Day:
    def __init__(self, group, sid, day, BBS):
        self.day = day
        self.BBS = BBS
        self.funcs = [Movement(group, sid, 'func_mov', day, motion)
                      for motion in FUNC_MV]
        self.indivs = [Movement(group, sid, 'indiv_mov', day, motion)
                       for motion in INDIV_MV]


class Movement:
    def __init__(self, *info):
        self.motion = info[4]
        self.path_list = sorted(glob.glob(os.path.join(DATA_PATH,
                                                       info[0],
                                                       f'R{info[1]:03}_*',
                                                       f'{info[2]}_{info[3]}',
                                                       f'{info[4]}_*.csv')))
        self.signals = [DataLoader(path) for path in self.path_list]


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

        emg, imu = self.data_load()
        self.emg = EMG(emg)
        self.imu = IMU(imu)

    def data_load(self):
        data = pd.read_csv(self.file_path)
        emg = self.emg_data(data)
        imu = self.imu_data(data)
        return emg, imu

    def emg_data(self, data):
        emg = data.loc[
            data['X[s]'].notnull(), ['EMG' in col for col in data.columns]]
        emg.columns =\
            [re.sub(': EMG \d+', '', i).replace(' ', '_') for i in emg.columns]
        emg = emg[630:-630] * 1000
        return emg.reset_index(drop=True)

    def imu_data(self, data):
        imu = data.loc[
            data['X[s].1'].notnull(),
            [True if re.search('ACC|GYRO', col) else False
             for col in data.columns]]
        imu.columns =\
            [re.sub(':| \d+', '', i).replace(' ', '_').replace('.', '_')
             for i in imu.columns]
        end_time = data['X[s]'].dropna().iloc[-1]
        imu = imu[data['X[s].1'].dropna() < end_time].reset_index(drop=True)
        imu = imu[74:-74]
        return imu.reset_index(drop=True)
