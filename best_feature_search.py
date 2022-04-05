from itertools import combinations, product
import pickle
import pandas as pd
import numpy as np


def feature_combinations(sensors, params):
    sensor_comb = []
    param_comb = []
    for i in range(1, 5):
        sensor_comb += list(combinations(sensors, i))
        param_comb += list(combinations(params, i))
    combs = list(product(*[sensor_comb, param_comb]))
    return combs


def data_helper(group, sensors, params):
    with open(f'../parameter/onset/{group}_onset_parameter.pkl', 'rb') as f:
        parameter = pickle.load(f)

    df = pd.DataFrame()
    for sid in parameter.keys():
        df2 = pd.DataFrame()
        for day in parameter[sid].keys():
            df3 = pd.DataFrame()
            for sensor in sensors:
                for param in params:
                    arr = parameter[sid][day][sensor][param]
                    df4 = pd.DataFrame(arr, columns=[f'{sensor}/{param}'])
                    df3 = pd.concat([df3, df4], axis=1).reset_index(drop=True)
            df2 = pd.concat([df2, df3], axis=0).reset_index(drop=True)
        df2['Subject'] = f'{group}_{sid.split("_")[-1]}'
        df2['BBS'] = parameter[sid][day]['BBS'] < 45

        df = pd.concat([df, df2], axis=0).reset_index(drop=True)
    return df
