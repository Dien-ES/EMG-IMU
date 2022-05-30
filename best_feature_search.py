from itertools import combinations, product
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, \
    confusion_matrix, make_scorer, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN, \
    BorderlineSMOTE, SMOTENC, SVMSMOTE, KMeansSMOTE

sns.set_style("whitegrid")

# test_sj = ['Healthy_4', 'Healthy_8', 'Healthy_12', 'Healthy_16', 'Healthy_20',
#            'Disabled_1',
#            'Disabled_5', 'Disabled_9', 'Disabled_15', 'Disabled_20']
#
# VAL0_SJ = ['Healthy_1', 'Healthy_5', 'Healthy_10', 'Disabled_4', 'Disabled_14']
# VAL1_SJ = ['Healthy_15', 'Healthy_19', 'Disabled_2', 'Disabled_6', 'Disabled_7']
# VAL2_SJ = ['Healthy_6', 'Healthy_7', 'Healthy_11', 'Disabled_10', 'Disabled_16']
# VAL3_SJ = ['Healthy_14', 'Healthy_17', 'Disabled_8', 'Disabled_3',
#            'Disabled_12']
# VAL4_SJ = ['Healthy_2', 'Healthy_3', 'Healthy_9', 'Disabled_11',
#            'Disabled_18']
# VAL5_SJ = ['Healthy_13', 'Healthy_18', 'Disabled_19', 'Disabled_21',
#            'Disabled_22']
# # VALS_SJ = [VAL0_SJ, VAL1_SJ, VAL2_SJ, VAL3_SJ, VAL4_SJ, VAL5_SJ]
# VALS_SJ = [VAL0_SJ + VAL1_SJ, VAL2_SJ + VAL3_SJ, VAL4_SJ + VAL5_SJ]

# 3fold
# test_sj = ['Healthy_4', 'Healthy_8', 'Healthy_12', 'Healthy_16', 'Healthy_20',
#            'Disabled_1', 'Disabled_7', 'Disabled_19', 'Disabled_11',
#            'Disabled_16', 'Disabled_20']
#
# VAL0_SJ = ['Healthy_1', 'Healthy_5', 'Healthy_10', 'Healthy_15', 'Healthy_19',
#            'Disabled_2', 'Disabled_4', 'Disabled_5', 'Disabled_3',
#            'Disabled_6']
# VAL1_SJ = ['Healthy_2', 'Healthy_6', 'Healthy_7', 'Healthy_11', 'Healthy_17',
#            'Disabled_8', 'Disabled_21', 'Disabled_14',
#            'Disabled_12']
# VAL2_SJ = ['Healthy_3', 'Healthy_9', 'Healthy_13', 'Healthy_14', 'Healthy_18',
#            'Disabled_9', 'Disabled_22', 'Disabled_10', 'Disabled_15',
#            'Disabled_18']
# VALS_SJ = [VAL0_SJ, VAL1_SJ, VAL2_SJ]

# 2fold
test_sj = [f'Healthy_{i}' for i in [3,4,8,9,12,13]] +\
          [f'Disabled_{i}' for i in [1,7,9,11,15,16,19,20]]

VAL0_SJ = [f'Healthy_{i}' for i in [1,5,10,14,15,16,19]] +\
          [f'Disabled_{i}' for i in [2,3,4,5,6,10]]
VAL1_SJ = [f'Healthy_{i}' for i in [2,6,7,11,17,18,20]] +\
          [f'Disabled_{i}' for i in [8,12,14,18,21,22]]
VALS_SJ = [VAL0_SJ, VAL1_SJ]


def feature_combinations(sensors, params):
    sensor_comb = []
    param_comb = []
    for i in range(1, len(params)+1):
        sensor_comb += list(combinations(sensors, i))
        param_comb += list(combinations(params, i))
    combs = list(product(*[sensor_comb, param_comb]))
    return combs


# COMBS = feature_combinations(['TFL', 'QF', 'GC', 'TA'],
#                              ['RMS_max', 'RMS_min', 'MVC', 'subMVC'])
COMBS = feature_combinations(['TFL', 'QF', 'GC', 'TA'],
                             ['RMS_max', 'RMS_min', 'MVC', 'subMVC',
                              'IMU_max', 'IMU_min'])


def data_helper(group, sensors, params, BBS_cut=29):
    with open(f'../parameter/onset/{group}_onset_parameter.pkl', 'rb') as f:
        parameter = pickle.load(f)

    df = pd.DataFrame()
    for sid in parameter.keys():
        df2 = pd.DataFrame()
        for day in parameter[sid].keys():
            df3 = pd.DataFrame()
            for sensor in sensors:
                for param in params:
                    if param[:3] == 'IMU':
                        for i, col in enumerate([f'{c}_{param[-3:]}' for c in
                                                 ['ACC_X', 'ACC_Y', 'ACC_Z',
                                                  'GYR_X', 'GYR_Y', 'GYR_Z']]):
                            arr = parameter['sid_1']['day_1']['TFL'][param][:,
                                  i]
                            df4 = pd.DataFrame(arr, columns=[f'{sensor}/{col}'])
                            df3 = pd.concat([df3, df4], axis=1).reset_index(
                                drop=True)
                    else:
                        arr = parameter[sid][day][sensor][param]
                        df4 = pd.DataFrame(arr, columns=[f'{sensor}/{param}'])
                        df3 = pd.concat([df3, df4], axis=1).reset_index(drop=True)
            df3['Day'] = day.split('_')[-1]
            df3['BBS'] = parameter[sid][day]['BBS'] <= BBS_cut
            df2 = pd.concat([df2, df3], axis=0).reset_index(drop=True)

        df2['Subject'] = f'{group}_{sid.split("_")[-1]}'
        # df2['BBS'] = parameter[sid]['day_1']['BBS'] <= BBS_cut
        df = pd.concat([df, df2], axis=0).reset_index(drop=True)
    return df


def cv_dataset(VALS_SJ, k):
    train_sj = VALS_SJ.copy()
    val_sj = train_sj.pop(k)
    train_sj = [i for sj in train_sj for i in sj]
    return train_sj, val_sj


def dataset_split(sensors, params, k):
    healthy_df = data_helper('Healthy', sensors, params)
    disabled_df = data_helper('Disabled', sensors, params)
    dataset = pd.concat([healthy_df, disabled_df]).reset_index(drop=True)
    dataset = dataset.loc[
        dataset.isnull().sum(1) != len(dataset.columns)].reset_index(drop=True)

    train_sj, val_sj = cv_dataset(VALS_SJ, k)
    train_bool, val_bool, test_bool = [], [], []
    for subject in dataset['Subject']:
        train_bool += [subject in train_sj]
        val_bool += [subject in val_sj]
        test_bool += [subject in test_sj]

    train_dataset = dataset.loc[train_bool].sample(frac=1).reset_index(
        drop=True)
    val_dataset = dataset.loc[val_bool].sample(frac=1).reset_index(drop=True)
    test_dataset = dataset.loc[test_bool].sample(frac=1).reset_index(drop=True)
    # print(f'<train_dataset>:\n{train_dataset["BBS"].value_counts()}\n')
    # print(f'<val_dataset>:\n{val_dataset["BBS"].value_counts()}\n')
    # print(f'<test_dataset>:\n{test_dataset["BBS"].value_counts()}\n')

    train_X = train_dataset.drop(['Subject', 'BBS', 'Day'], axis=1)
    train_Y = train_dataset['BBS'].astype(int).values
    val_X = val_dataset.drop(['Subject', 'BBS', 'Day'], axis=1)
    val_Y = val_dataset['BBS'].astype(int).values
    test_X = test_dataset.drop(['Subject', 'BBS', 'Day'], axis=1)
    test_Y = test_dataset['BBS'].astype(int).values
    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def preprocessing(train_X, val_X, test_X):
    ppl = Pipeline([
        ('imputer', KNNImputer(n_neighbors=3)),
        ('scaler', MinMaxScaler())
    ])

    ppl.fit(train_X)
    train_X = ppl.transform(train_X)
    val_X = ppl.transform(val_X)
    test_X = ppl.transform(test_X)
    return train_X, val_X, test_X


def metric_result(test_Y, test_pred, test_prob):
    try:
        TN, FP, FN, TP = confusion_matrix(test_Y, test_pred).ravel()
        sen = TP / (TP + FN)
        spe = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (FN + TN)

    except:
        sen, spe, PPV, NPV = 0, 0, 0, 0

    acc = accuracy_score(test_Y, test_pred)
    f1 = f1_score(test_Y, test_pred)
    mcc = matthews_corrcoef(test_Y, test_pred)

    try:
        AUC = roc_auc_score(test_Y, test_prob[:, 1])
    except:
        print('ValueError: Only one class present in y_true.'
              ' ROC AUC score is not defined in that case.')
        AUC = 0

    return acc, f1, mcc, AUC, sen, spe, PPV, NPV


def roc_auc_result(test_Y, test_prob):
    fprs, tprs, thresholds = roc_curve(test_Y, test_prob[:, 1])
    auc = roc_auc_score(test_Y, test_prob[:, 1])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1])
    plt.plot(fprs, tprs, label=f'AUC = {auc: 0.2f}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.ylabel('TPR', fontsize=20)
    plt.xlabel('FPR', fontsize=20)
    plt.tick_params(labelsize=15)
    # plt.title(f'{mv}', size=20)
    plt.legend(loc='lower right', fontsize=15)
    return auc


def modeling(COMBS, result='valid', oversampler=None, k_fold=2):
    total_result = pd.DataFrame()
    for sensors, params in tqdm(COMBS):
        results = []
        for k in range(k_fold):
            train_X, train_Y, val_X, val_Y, test_X, test_Y = dataset_split(
                sensors, params, k)
            train_X, val_X, test_X = preprocessing(train_X, val_X, test_X)
            if oversampler:
                try:
                    train_X, train_Y = oversampler.fit_resample(train_X,
                                                                train_Y)
                except:
                    pass

            rf = RandomForestClassifier(n_estimators=100, random_state=101)
            rf.fit(train_X, train_Y)

            if result == 'test':
                pred = rf.predict(test_X)
                prob = rf.predict_proba(test_X)
                results += [metric_result(test_Y, pred, prob)]
            else:
                pred = rf.predict(val_X)
                prob = rf.predict_proba(val_X)
                results += [metric_result(val_Y, pred, prob)]

        result_df = pd.DataFrame(np.mean(results, axis=0).reshape(1, -1),
                                 columns=['Accuracy', 'F-score', 'MCC', 'AUC',
                                          'Sensitivity', 'Specificity',
                                          'PPV', 'NPV'])
        result_df['Sensors'] = [sensors]
        result_df['Parameters'] = [params]
        result_df['train False:True'] = [
            tuple(np.unique(train_Y, return_counts=True)[1])]
        if result == 'test':
            result_df['test False:True'] = [
                tuple(np.unique(test_Y, return_counts=True)[1])]
        else:
            result_df['val False:True'] = [
                tuple(np.unique(val_Y, return_counts=True)[1])]

        total_result = pd.concat([total_result, result_df], axis=0).reset_index(
            drop=True)

    total_result = pd.concat([total_result.iloc[:, -4:],
                             total_result.iloc[:, :-4]], axis=1)
    total_result = total_result.sort_values('MCC', ascending=False).\
        reset_index(drop=True)
    return total_result
