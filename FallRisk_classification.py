from EMG_analysis import *

COL_IDX = [2, 3, 4, 5, 8, 9, 10, 11]


def data_helper(params, mv_list):
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
