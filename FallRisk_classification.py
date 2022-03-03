from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef,\
    confusion_matrix, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import Normalizer

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
                          [f'RMS/{c}' for c in np.array(COLUMNS)[COL_IDX]] + \
                          [f'MVC/{c}' for c in np.array(COLUMNS)[COL_IDX]] + \
                          [f'subMVC/{c}' for c in np.array(COLUMNS)[COL_IDX]] + \
                          amp_colnames('RMS') +\
                          amp_colnames('MVC') +\
                          amp_colnames('subMVC')
                else:
                    df2 = pd.DataFrame(group +
                                       rms + mvc + submvc).T
                    df2.columns = ['Group'] + \
                          [f'RMS/{c}' for c in np.array(COLUMNS)[COL_IDX]] + \
                          [f'MVC/{c}' for c in np.array(COLUMNS)[COL_IDX]] + \
                          [f'subMVC/{c}' for c in np.array(COLUMNS)[COL_IDX]]

                df1 = pd.concat([df1, df2])

        df1['Movement'] = mv

        total_df = pd.concat([total_df, df1])
    return total_df.reset_index(drop=True)


def amp_colnames(param):
    colnames = []
    for i in [1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0]:
        colnames += [f'{param}_AMP({i})/{c}'
                     for c in np.array(COLUMNS)[COL_IDX]]
    return colnames


def train_test_dataset(mv_df, mv):
    df = mv_df[mv_df['Movement'] == mv].reset_index(drop=True)
    df = df.drop('Movement', axis=1)

    X = df.drop('Group', axis=1)
    Y = df['Group'].astype(int).values

    train_X, test_X, train_Y, test_Y = train_test_split(X, Y,
                                                        test_size=0.25,
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


def metric_result(test_Y, test_pred):
    TN, FP, FN, TP = confusion_matrix(test_Y, test_pred).ravel()
    acc = accuracy_score(test_Y, test_pred)
    f1 = f1_score(test_Y, test_pred)
    mtc = matthews_corrcoef(test_Y, test_pred)
    sen = TP/(TP+FN)
    spe = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(FN+TN)
    return acc, f1, mtc, sen, spe, PPV, NPV


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
    cv_result_melt = cv_result_df.melt(value_vars=[f'split{i}_test_score' for i in range(5)],
                                       value_name='Score', var_name='CV')
    cv_result_melt['n_features'] = list(range(1, 25))*5

    n_features = rfecv.n_features_
    score = cv_result_df['mean_test_score'].max()

    plt.figure(figsize=(6, 6))

    sns.lineplot(data=cv_result_melt,
                 x='n_features',
                 y='Score')
    plt.vlines(x=n_features, ymin=0, ymax=1, ls='--', colors='red',
               label=f'n_features = {n_features}\nScore = {score: 0.2f}')
    plt.legend(loc='lower left', fontsize=15)
    plt.ylabel('Score (MCC)', fontsize=20)
    plt.xlabel('n_features', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.ylim(0.3,1)
    plt.title(f'{mv}', size=20)
    plt.show()
