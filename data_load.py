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
D_INFO = pd.read_csv(os.path.join(DATA_PATH, 'Disabled_info.csv'))


class Subject:
    def __init__(self, group, sid):
        self.group = group
        self.sid = int(sid.split('_')[-1])
        self.af_side = None
        if group == 'Disabled':
            self.af_side = D_INFO.query(f'Sid=={self.sid}')['AffectedSide'].values[0]
        self.days = self.load(group, self.sid, self.af_side)

    def load(self, group, sid, af_side):
        days = {}
        for day in ['day_1', 'day_2']:
            d = Day(group, sid, day, af_side)
            if (d.funcs is not None) | (d.indivs is not None):
                days[day] = d

        if len(days) > 0:
            return days
        else:
            return None


class Day:
    def __init__(self, group, sid, day, af_side):
        self.BBS = 60
        self.af_side = af_side
        self.day = int(day.split('_')[-1])
        if group == 'Disabled':
            self.BBS = D_INFO.query(f'Sid=={sid}')[f'BBS_{self.day}'].values[0]

        self.funcs = self.load(group, sid, 'func_mov', FUNC_MV, self.day, af_side)
        self.indivs = self.load(group, sid, 'indiv_mov', INDIV_MV, self.day, af_side)

    def load(self, group, sid, movement, MV, day, af_side):
        moves = {}
        for motion in MV:
            move = self.movement(group, sid, movement, day, motion, af_side)
            # affected_side switching
            if move is not None:
                if af_side == 'Left':
                    if motion[0] == 'l':
                        motion = f'r{motion[1:]}'
                    elif motion[0] == 'r':
                        motion = f'l{motion[1:]}'

                moves[motion] = move

        if len(moves) > 0:
            return moves
        else:
            return None

    def movement(self, *info):
        path_list = sorted(glob.glob(os.path.join(DATA_PATH,
                                                  info[0],
                                                  f'R{info[1]:03}_*',
                                                  f'{info[2]}_{info[3]}',
                                                  f'{info[4]}*.csv')))
        if len(path_list) > 0:
            return [DataLoader(path, info[0], info[1], info[3], info[4], info[5])
                    for path in path_list]
        else:
            return None


class DataLoader:
    def __init__(self, file_path, group, sid, day, motion, af_side):
        self.file_path = file_path
        self.group = group
        self.sid = sid
        self.day = day
        self.motion = motion
        self.af_side = af_side
        emg, imu = self.data_load(group)
        self.emg = EMG(emg)
        self.imu = IMU(imu)

    def data_load(self, group):
        data = pd.read_csv(self.file_path)
        emg = self.emg_data(data, group, self.af_side)
        imu = self.imu_data(data, group, self.af_side)
        return emg, imu

    def emg_data(self, data, group, af_side):
        emg = data.loc[
            data['X[s]'].notnull(), ['EMG' in col for col in data.columns]]
        emg.columns = \
            [re.sub(': EMG \d+', '', i).replace(' ', '_') for i in emg.columns]
        emg = emg[630:-630] * 1000
        emg = emg.reset_index(drop=True).copy()

        if (group == 'Disabled') & (af_side == 'Left'):
            columns = emg.columns
            a_emg = pd.concat([emg.iloc[:, 6:], emg.iloc[:, :6]], axis=1)
            a_emg.columns = columns
            return a_emg
        return emg

    def imu_data(self, data, group, af_side):
        imu = data.loc[
            data['X[s].1'].notnull(),
            [True if re.search('ACC|GYRO', col) else False
             for col in data.columns]]
        imu.columns = \
            [re.sub(':| \d+', '', i).replace(' ', '_').replace('.', '_')
             for i in imu.columns]
        end_time = data['X[s]'].dropna().iloc[-1]
        imu = imu[data['X[s].1'].dropna() < end_time].reset_index(drop=True)
        imu = imu[74:-74]
        imu = imu.reset_index(drop=True).copy()

        if (group == 'Disabled') & (af_side == 'Left'):
            columns = imu.columns
            a_imu = pd.concat([imu.iloc[:, 36:], imu.iloc[:, :36]], axis=1)
            a_imu.columns = columns
            return a_imu
        return imu
