from EMG_analysis import *


def data_helper(params, mv_list):
    total_df = pd.DataFrame()
    for idx, mv in enumerate(mv_list):
        df1 = pd.DataFrame()
        for param in params:
            if param.info['motion'] == mv:
                group = [param.info['BBS'] < 45]
                rms = list(np.nanmax(param.rms.data, axis=0))
                mvc = list(np.nanmax(param.mvc.data, axis=0))
                submvc = list(np.nanmax(param.submvc.data, axis=0))
                df2 = pd.DataFrame(group + rms + mvc + submvc).T
                df2.columns = ['Group'] + [f'RMS/{c}' for c in COLUMNS] + [f'MVC/{c}' for c in COLUMNS] + [f'subMVC/{c}' for c in COLUMNS]
                df1 = pd.concat([df1, df2])

        df1 = df1.iloc[:, [0,
                           3, 4, 5, 6, 9, 10, 11, 12,
                           15, 16, 17, 18, 21, 22, 23, 24,
                           27, 28, 29, 30, 33, 34, 35, 36]]
        df1['Movement'] = mv

        total_df = pd.concat([total_df, df1])
    return total_df.reset_index(drop=True)

