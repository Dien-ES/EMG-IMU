from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, \
    confusion_matrix, make_scorer, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import Normalizer
import re
from sklearn.svm import SVC

from EMG_analysis import *

COL_IDX = [2, 3, 4, 5, 8, 9, 10, 11]


def data_helper(params, mv_list, is_amp=False):
    total_df = pd.DataFrame()
    for idx, mv in enumerate(mv_list):
        df1 = pd.DataFrame()

        for param in params:
            if param.info['motion'] == mv:
                group = [param.info['BBS'] < 45]
                rms = list(np.nanmax(param.rms.data[:, COL_IDX], axis=0))
                mvc = list(np.nanmax(param.mvc.data[:, COL_IDX], axis=0))
                submvc = list(np.nanmax(param.submvc.data[:, COL_IDX],
                                        axis=0))

                if is_amp:
                    rms_amp = list(param.rms_amp[:, COL_IDX].reshape(-1))
                    mvc_amp = list(param.mvc_amp[:, COL_IDX].reshape(-1))
                    submvc_amp = list(param.submvc_amp[:, COL_IDX].reshape(-1))

                    df2 = pd.DataFrame(group +
                                       rms + mvc + submvc +
                                       rms_amp + mvc_amp + submvc_amp).T
                    df2.columns = ['Group'] + \
                                  [f'RMS/{c}' for c in
                                   np.array(COLUMNS)[COL_IDX]] + \
                                  [f'MVC/{c}' for c in
                                   np.array(COLUMNS)[COL_IDX]] + \
                                  [f'subMVC/{c}' for c in
                                   np.array(COLUMNS)[COL_IDX]] + \
                                  amp_colnames('RMS') + \
                                  amp_colnames('MVC') + \
                                  amp_colnames('subMVC')
                else:
                    df2 = pd.DataFrame(group +
                                       rms + mvc + submvc).T
                    df2.columns = ['Group'] + \
                                  [f'RMS/{c}' for c in
                                   np.array(COLUMNS)[COL_IDX]] + \
                                  [f'MVC/{c}' for c in
                                   np.array(COLUMNS)[COL_IDX]] + \
                                  [f'subMVC/{c}' for c in
                                   np.array(COLUMNS)[COL_IDX]]

                df1 = pd.concat([df1, df2])

        df1['Movement'] = mv

        total_df = pd.concat([total_df, df1])
    return total_df.reset_index(drop=True)


def rms_data_helper(params, mv_list):
    total_df = pd.DataFrame()
    for mv in mv_list:
        df1 = pd.DataFrame()
        for param in params:
            if param.info['motion'] == mv:
                rms = list(np.nanmax(param.rms.data[:, COL_IDX], axis=0))
                df2 = pd.DataFrame([param.info['group']] +\
                                   [param.info['sid']] +\
                                   [param.info['BBS']] +\
                                   [param.info['day']] +\
                                   [param.info['motion']] +\
                                   rms).T
                df2.columns = ['Group', 'Subject_ID', 'BBS', 'Day', 'Movement'] +\
                              list(np.array(COLUMNS)[COL_IDX])
                df1 = pd.concat([df1, df2])
        total_df = pd.concat([total_df, df1])

    total_df = total_df.reset_index(drop=True)
    total_df = columns_astype(total_df)
    return total_df


def columns_astype(df):
    df['Group'] = df['Group'].astype(str)
    df['Subject_ID'] = df['Subject_ID'].astype(int)
    df['BBS'] = df['BBS'].astype(int)
    df['Day'] = df['Day'].astype(int)
    df['Movement'] = df['Movement'].astype(str)
    for col in list(np.array(COLUMNS)[COL_IDX]):
        df[col] = df[col].astype(float)
    return df


def amp_colnames(param):
    colnames = []
    for i in [1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0]:
        colnames += [f'{param}_AMP({i})/{c}'
                     for c in np.array(COLUMNS)[COL_IDX]]
    return colnames


def train_test_dataset(mv_df, mv, sensor=None, is_MVC=None, test_size=0.25):
    df = mv_df[mv_df['Movement'] == mv].reset_index(drop=True)
    df = df.drop('Movement', axis=1)

    X = df.drop('Group', axis=1)
    if sensor:
        if is_MVC:
            X = X.loc[:, [True if re.match(f'((RMS)|(MVC))/{sensor}', col)
                          else False for col in X.columns]]
        else:
            X = X.loc[:, [True if re.match(f'((RMS)|(subMVC))/{sensor}', col)
                          else False for col in X.columns]]

    Y = df['Group'].astype(int).values

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y,
                                                        test_size=test_size,
                                                        stratify=Y,
                                                        random_state=101)
    feature_names = train_X.columns

    imputer = KNNImputer(n_neighbors=3, weights='uniform')
    train_X = imputer.fit_transform(train_X)
    test_X = imputer.transform(test_X)
    normalizer = Normalizer()
    train_X = normalizer.fit_transform(train_X)
    test_X = normalizer.transform(test_X)

    train_X = pd.DataFrame(train_X, columns=feature_names)
    test_X = pd.DataFrame(test_X, columns=feature_names)
    return train_X, test_X, train_Y, test_Y


def metric_result(test_Y, test_pred, test_prob, mv=None):
    TN, FP, FN, TP = confusion_matrix(test_Y, test_pred).ravel()
    acc = accuracy_score(test_Y, test_pred)
    f1 = f1_score(test_Y, test_pred)
    mtc = matthews_corrcoef(test_Y, test_pred)
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (FN + TN)
    if mv:
        AUC = roc_auc_result(test_Y, test_prob, mv)
    else:
        AUC = roc_auc_score(test_Y, test_prob[:, 1])

    return acc, f1, mtc, sen, spe, PPV, NPV, AUC, test_Y.shape[0]


def roc_auc_result(test_Y, test_prob, mv):
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
    plt.title(f'{mv}', size=20)
    plt.legend(loc='lower right', fontsize=15)
    return auc


def feature_selection_RFECV(train_X, train_Y, k, min_features):
    rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                random_state=101)
    rfecv = RFECV(estimator=rf,
                  cv=StratifiedKFold(k, shuffle=True, random_state=101),
                  scoring=make_scorer(matthews_corrcoef),
                  min_features_to_select=min_features)
    rfecv.fit(train_X, train_Y)

    return rfecv


def feature_topk(rfecv, k):
    return rfecv.feature_names_in_[rfecv.ranking_ <= k]


def RFECV_plot(rfecv, mv):
    cv_result_df = pd.DataFrame(rfecv.cv_results_)
    cv_result_melt = cv_result_df.melt(value_vars=[f'split{i}_test_score'
                                                   for i in range(5)],
                                       value_name='Score', var_name='CV')
    cv_result_melt['n_features'] = list(range(1, 25)) * 5

    n_features = rfecv.n_features_
    score = cv_result_df['mean_test_score'].max()

    plt.figure(figsize=(6, 6))

    sns.lineplot(data=cv_result_melt,
                 x='n_features',
                 y='Score')
    plt.vlines(x=n_features, ymin=0, ymax=1, ls='--', colors='red',
               label=f'n_features = {n_features}\nMCC = {score: 0.2f}')
    plt.legend(loc='lower left', fontsize=15)
    plt.ylabel('Score (MCC)', fontsize=20)
    plt.xlabel('n_features', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim(0, 1)
    plt.xlim(1, 24)
    plt.title(f'{mv}', size=20)
    plt.show()


def feature_importance(rfecv, mv, train_X, train_Y, test_X, test_Y):
    topk = feature_topk(rfecv, k=1)
    topk_train_X = train_X[topk]
    topk_test_X = test_X[topk]

    rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                random_state=101)
    rf.fit(topk_train_X, train_Y)

    ft_importance_values = rf.feature_importances_
    ft_series = pd.Series(ft_importance_values, index=topk_test_X.columns)
    ft_series = ft_series.sort_values(ascending=False)

    test_pred = rf.predict(topk_test_X)
    test_prob = rf.predict_proba(topk_test_X)
    results = metric_result(test_Y, test_pred, test_prob, mv)

    return results, ft_series


def feature_importance_plot(ft_series, mv):
    plt.figure(figsize=(ft_series.shape[0], 5))
    plt.title(f'{mv}', size=20)
    sns.barplot(x=ft_series.index, y=ft_series)
    plt.tick_params(labelsize=12)
    plt.xticks(rotation=45)
    plt.xlabel('feature_importance', fontsize=15)
    plt.show()


def movement_modeling_result(params, MV_list):
    mv_df = data_helper(params, MV_list)

    result_columns = ['MV', 'Acc', 'F', 'MCC', 'Sen',
                      'Spe', 'PPV', 'NPV', 'AUC', 'No. test']
    result_df = pd.DataFrame(columns=result_columns)

    feature_list = ['RMS/R_TFL', 'RMS/R_QF', 'RMS/R_GC', 'RMS/R_TA',
                    'RMS/L_TFL', 'RMS/L_QF', 'RMS/L_GC', 'RMS/L_TA',
                    'MVC/R_TFL', 'MVC/R_QF', 'MVC/R_GC', 'MVC/R_TA',
                    'MVC/L_TFL', 'MVC/L_QF', 'MVC/L_GC', 'MVC/L_TA',
                    'subMVC/R_TFL', 'subMVC/R_QF', 'subMVC/R_GC', 'subMVC/R_TA',
                    'subMVC/L_TFL', 'subMVC/L_QF', 'subMVC/L_GC', 'subMVC/L_TA']
    fi_df = pd.DataFrame(columns=feature_list)
    for mv in MV_list:
        train_X, test_X, train_Y, test_Y = train_test_dataset(mv_df, mv)
        rfecv = feature_selection_RFECV(train_X, train_Y, k=5, min_features=1)

        results, ft_series = feature_importance(rfecv, mv,
                                                train_X, train_Y,
                                                test_X, test_Y)
        RFECV_plot(rfecv, mv)
        feature_importance_plot(ft_series, mv)

        new_df = pd.DataFrame([mv] + list(results)).T
        new_df.columns = result_columns
        result_df = pd.concat([result_df, new_df], axis=0). \
            reset_index(drop=True)
        fi_df = pd.concat([fi_df, pd.DataFrame(ft_series).T], axis=0). \
            reset_index(drop=True)

    return result_df, fi_df


def rf_result(train_X, train_Y, test_X, test_Y):
    rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                random_state=101)
    rf.fit(train_X, train_Y)

    test_pred = rf.predict(test_X)
    test_prob = rf.predict_proba(test_X)
    results = metric_result(test_Y, test_pred, test_prob)
    return results


def sensor_comparison_result(params, MV_list, is_MVC, title):
    mv_df = data_helper(params, MV_list)
    result_columns = ['MV', 'Sensor', 'Acc', 'F', 'MCC', 'Sen',
                      'Spe', 'PPV', 'NPV', 'AUC']
    result_df = pd.DataFrame(columns=result_columns)
    for mv in tqdm(MV_list):
        for sensor in ['R_TFL', 'R_QF', 'R_GC', 'R_TA',
                       'L_TFL', 'L_QF', 'L_GC', 'L_TA']:
            train_X, test_X, train_Y, test_Y = train_test_dataset(mv_df, mv,
                                                                  sensor,
                                                                  is_MVC=is_MVC)
            results = rf_result(train_X, train_Y, test_X, test_Y)
            new_df = pd.DataFrame([mv, sensor] + list(results)[:-1]).T
            new_df.columns = result_columns
            result_df = pd.concat([result_df, new_df], axis=0). \
                reset_index(drop=True)

    result_df_melt = result_df.melt(id_vars=['Sensor', 'MV'],
                                    var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.pointplot(data=result_df_melt,
                  x='Sensor',
                  y='Score',
                  hue='Metric')
    plt.tick_params(labelsize=15)
    plt.legend(fontsize=15, bbox_to_anchor=(1.05, 1))
    plt.ylabel('Score', fontsize=20)
    plt.xlabel('Sensor', fontsize=20)
    plt.title(title, size=20)
    return result_df
