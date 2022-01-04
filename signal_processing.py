import os
import numpy as np
import pandas as pd
import glob
import re

DATA_PATH = '../../../../Database/EMG_IMU'
SENSORS = ['TP', 'LD', 'TFL', 'QF', 'GC', 'TA']
FUNC_MV = ['back_extension', 'back_flexion',
           'sit_to_stand', 'supine_to_sit', 'gait']
INDIV_MV = []
for side in ['r', 'l']:
    for task in ['habd', 'ke', 'apf', 'adf']:
        for power in ['max', 'min']:
            INDIV_MV += [f'{side}_{task}_{power}']


class Signal:
    def __init__(self, is_prep, *file_path):
        self.file_path = file_path
        self.emg, self.imu = self.data_load()
        if is_prep:
            self.emg.preprocessing()
            self.imu.preprocessing()

    def data_load(self):
        path = glob.glob(os.path.join(DATA_PATH, *self.file_path))[0]
        data = pd.read_csv(path)
        emg = self.emg_data(data)
        imu = self.imu_data(data)
        return EMG(emg), IMU(imu)

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
            [True if re.search("ACC|GYRO", col) else False
             for col in data.columns]]
        imu.columns =\
            [re.sub(':| \d+', '', i).replace(' ', '_').replace('.', '_')
             for i in imu.columns]
        end_time = data["X[s]"].dropna().iloc[-1]
        imu = imu[data["X[s].1"].dropna() < end_time].reset_index(drop=True)
        imu = imu[74:-74]
        return imu.reset_index(drop=True)


class EMG:
    def __init__(self, data, fs=1259):
        self.data = data
        self.fs = fs

    def preprocessing(self):
        # ??

        self.data = None


class IMU:
    def __init__(self, data, fs=148):
        self.data = data
        self.fs = fs

    def preprocessing(self):
        # ??

        self.data = None
