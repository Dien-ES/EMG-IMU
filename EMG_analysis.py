import pickle
from statannot import add_stat_annotation
import pingouin as pg
from scipy.stats import chisquare, chi2_contingency, pearsonr

from EMG_parameter import *


def mvc_submvc_concat(params, mv, group_1='BBS', group_2=None):
    mvc_df = melting_df(params, mv, 'MVC', group_1, group_2)
    submvc_df = melting_df(params, mv, 'subMVC', group_1, group_2)
    df = pd.concat([mvc_df, submvc_df['subMVC']], axis=1)
    df_melt = df.melt(id_vars=['Group', 'Sensor', 'Side'],
                      var_name='Contraction')
    df_melt['value'] = df_melt['value'].astype(float)
    return df_melt


def melting_df(params, mv, contr, group_1='BBS', group_2=None):
    df = pd.DataFrame(columns=['Group'] + COLUMNS)
    for param in params:
        if param.info['motion'] == mv:
            info = []
            if contr == 'MVC':
                value = list(np.nanmax(param.mvc.data, axis=0))
            elif contr == 'subMVC':
                value = list(np.nanmax(param.submvc.data, axis=0))
            else:
                value = list(np.nanmax(param.rms.data, axis=0))

            info += [param.info['group'] == 'Disabled']
            info += [param.info['BBS'] < 45]
            info += [param.info['day'] == 2]
            new_df = pd.DataFrame(info + value).T
            new_df.columns = ['is_disabled', 'is_fallRisk',
                              'is_secondDay'] + COLUMNS
            df = pd.concat([df, new_df])

    if group_1 == 'BBS':
        df['Group'] = df['is_fallRisk']
    elif (group_1 == 'Healthy_1') & (group_2 == 'Disabled_1'):
        df = df.loc[df['is_secondDay'] == False].reset_index(drop=True)
        df['Group'] = df['is_disabled']
    else:
        df = df.loc[df['is_disabled']].reset_index(drop=True)
        df['Group'] = df['is_secondDay']

    df = df.drop(['is_disabled', 'is_fallRisk', 'is_secondDay'], axis=1)
    df_melt = df.melt(id_vars='Group', var_name='Sensor', value_name=contr)
    df_melt['Side'] = df_melt['Sensor'].str.split('_').str[0]
    df_melt['Sensor'] = df_melt['Sensor'].str.split('_').str[1]
    df_melt[contr] = df_melt[contr].astype(float)
    df_melt['Group'] = df_melt['Group'].astype(str)
    return df_melt


def amplitude_melting_df(params, mv, contr, group_1='BBS', group_2=None):
    df = pd.DataFrame(columns=['Group'] + COLUMNS)
    for param in params:
        if param.info['motion'] == mv:
            for i, time in enumerate([1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0]):
                info = []
                if contr == 'MVC':
                    value = list(param.mvc_amp[i].astype(int))
                elif contr == 'subMVC':
                    value = list(param.submvc_amp[i].astype(int))
                else:
                    value = list(param.rms_amp[i].astype(int))

                info += [param.info['group'] == 'Disabled']
                info += [param.info['BBS'] < 45]
                info += [param.info['day'] == 2]
                info += [time]
                new_df = pd.DataFrame(info + value).T
                new_df.columns = ['is_disabled', 'is_fallRisk',
                                  'is_secondDay', 'Times'] + COLUMNS
                df = pd.concat([df, new_df])

    if group_1 == 'BBS':
        df['Group'] = df['is_fallRisk']
    elif (group_1 == 'Healthy_1') & (group_2 == 'Disabled_1'):
        df = df.loc[df['is_secondDay'] == False].reset_index(drop=True)
        df['Group'] = df['is_disabled']
    else:
        df = df.loc[df['is_disabled']].reset_index(drop=True)
        df['Group'] = df['is_secondDay']

    df_melt = pd.DataFrame()
    for row, sensor in enumerate(['TFL', 'QF', 'GC', 'TA']):
        for col, side in enumerate(['L', 'R']):
            df_crosstab = pd.crosstab([df.Group, df.Times],
                                      df[f'{side}_{sensor}'])
            df_crosstab = df_crosstab.reset_index()
            if 0 not in df_crosstab.columns:
                df_crosstab[0] = 0
            if 1 not in df_crosstab.columns:
                df_crosstab[1] = 0
            df_crosstab.columns = ['Group', 'Times', 'Minus', 'Plus']
            df_crosstab['Side'] = side
            df_crosstab['Sensor'] = sensor

            df_melt = pd.concat([df_melt, df_crosstab]).reset_index(drop=True)
    df_melt['Times'] = df_melt['Times'].astype(float)
    return df_melt


def Disabled_BBS_RMS_variation(params, mv):
    df = pd.DataFrame(columns=COLUMNS)
    for param in params:
        if param.info['motion'] == mv:
            info = []
            value = list(np.nanmax(param.rms.data, axis=0))

            info += [param.info['group'] == 'Disabled']
            info += [param.info['BBS']]
            info += [param.info['day']]
            info += [param.info['sid']]
            new_df = pd.DataFrame(info + value).T
            new_df.columns = ['is_disabled', 'BBS',
                              'Day', 'Sid'] + COLUMNS
            df = pd.concat([df, new_df])

    df = df.loc[df['is_disabled']].reset_index(drop=True)
    df = df.drop(['is_disabled'], axis=1)
    if mv in FUNC_MV:
        df = df.pivot_table(index=['Sid', 'Day'], aggfunc='mean').reset_index()

    counts = df['Sid'].value_counts()
    cond = [sid in counts[counts == 2].index for sid in df['Sid']]
    df = df.loc[cond].reset_index(drop=True)

    df2 = df.loc[df['Day'] == 2, 'BBS'].reset_index(drop=True)
    df1 = df.loc[df['Day'] == 1, 'BBS'].reset_index(drop=True)
    df_var = df2 - df1
    for row, sensor in enumerate(['TFL', 'QF', 'GC', 'TA']):
        for col, side in enumerate(['L', 'R']):
            df2 = df.loc[df['Day'] == 2, f'{side}_{sensor}'].reset_index(drop=True)
            df1 = df.loc[df['Day'] == 1, f'{side}_{sensor}'].reset_index(drop=True)
            df_var = pd.concat([df_var, df2-df1], axis=1).reset_index(drop=True)

    return df_var


def mvc_boxplot(df_melt, mv, ylim=None):
    df_pivot = df_melt.pivot_table(index=['Side', 'Sensor'],
                                   columns=['Contraction', 'Group'],
                                   aggfunc='count')

    fig, axes = plt.subplots(4, 2, figsize=(6, 12),
                             sharey=True,
                             constrained_layout=True)

    for row, sensor in enumerate(['TFL', 'QF', 'GC', 'TA']):
        for col, side in enumerate(['L', 'R']):
            ax = axes[row, col]
            n1, n2, n3, n4 =\
                df_pivot[df_pivot.index == (side, sensor)].values[0]
            data = df_melt[
                (df_melt['Side'] == side) & (df_melt['Sensor'] == sensor)]
            data = data.dropna().reset_index(drop=True)
            sns.boxplot(ax=ax,
                        data=data,
                        x='Contraction', y='value',
                        hue='Group', palette='Set2',
                        # order=['False', 'True'],
                        showfliers=False)

            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.set_xticklabels([f'MVC\n({n1}:{n2})', f'subMVC\n({n3}:{n4})'])
            ax.set_ylabel(f'{side}_{sensor}', fontsize=20)
            ax.set_xlabel('')
            ax.tick_params(labelsize=15)
            ax.legend([], [], frameon=False)
            add_stat_annotation(ax,
                                data=data,
                                x='Contraction',
                                y='value',
                                hue='Group',
                                box_pairs=[(('MVC', 'False'), ('MVC', 'True')),
                                           (('subMVC', 'False'), ('subMVC', 'True'))],
                                test='Mann-Whitney',  # -gt, -ls
                                loc='outside',
                                text_format='star',
                                fontsize='large',
                                verbose=2,
                                comparisons_correction=None
                                )
    fig.suptitle(f'{mv}', fontweight="bold", fontsize=20)


def boxplot(df_melt, mv, contr, ylim=None):
    df_pivot = df_melt.pivot_table(index=['Side', 'Sensor'],
                                   columns=['Group'],
                                   aggfunc='count')

    fig, axes = plt.subplots(4, 2, figsize=(6, 12),
                             sharey=True,
                             constrained_layout=True)

    for row, sensor in enumerate(['TFL', 'QF', 'GC', 'TA']):
        for col, side in enumerate(['L', 'R']):
            ax = axes[row, col]
            n1, n2 = df_pivot[df_pivot.index == (side, sensor)].values[0]
            data = df_melt[
                (df_melt['Side'] == side) & (df_melt['Sensor'] == sensor)]
            data = data.dropna().reset_index(drop=True)
            sns.boxplot(ax=ax,
                        data=data,
                        x='Group',
                        y=contr,
                        palette='Set2',
                        order=['False', 'True'],
                        showfliers=False)

            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.set_xlabel(f'({n1}:{n2})', fontsize=14)
            ax.set_ylabel(f'{side}_{sensor}', fontsize=20)

            ax.tick_params(labelsize=15)
            ax.legend([], [], frameon=False)
            add_stat_annotation(ax,
                                data=data,
                                x='Group',
                                y=contr,
                                box_pairs=[('False', 'True')],
                                test='Mann-Whitney',  # -gt, -ls
                                loc='outside',
                                text_format='star',
                                fontsize='large',
                                verbose=2,
                                comparisons_correction=None
                                )

    fig.suptitle(f'{mv}', fontweight="bold", fontsize=20)


def amplitude_barplot(df_melt, mv, ylim=1.0):
    fig, axes = plt.subplots(4, 2, figsize=(6, 12),
                             sharey=True,
                             constrained_layout=True)

    for row, sensor in enumerate(['TFL', 'QF', 'GC', 'TA']):
        for col, side in enumerate(['L', 'R']):
            xticklabels = []
            for time in [1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0]:
                xo, xe = df_melt.loc[(df_melt['Side'] == side) & (
                            df_melt['Sensor'] == sensor) & (
                                                 df_melt['Times'] == time), [
                                         'Minus', 'Plus']].values
                if ((xo[0] == 0) & (xe[0] == 0)) | ((xo[1] == 0) & (xe[1] == 0)):
                    xticklabels += [f'{time}\n']
                else:
                    _, p, _, _ = chi2_contingency([xo, xe])
                    xticklabels += [f'{time}\n{pvalue_cut(p)}']

            ax = axes[row, col]
            data = df_melt[
                (df_melt['Side'] == side) & (df_melt['Sensor'] == sensor)]
            data = data.dropna().reset_index(drop=True)

            sns.barplot(ax=ax,
                        data=data,
                        x='Times',
                        y=data['Plus']/(data['Plus'] + data['Minus']),
                        hue='Group', palette='Set2')

            if ylim is not None:
                ax.set_ylim(0, ylim)
            ax.set_ylabel(f'{side}_{sensor}', fontsize=20)
            ax.set_xlabel('')
            ax.set_xticklabels(xticklabels)
            ax.tick_params(labelsize=10)
            ax.legend([], [], frameon=False)
            fig.suptitle(f'{mv}', fontweight="bold", fontsize=20)


def correlation_plot(df_var, mv):
    fig, axes = plt.subplots(4, 2, figsize=(6, 12),
                             sharey=True, sharex=True,
                             constrained_layout=True)
    for row, sensor in enumerate(['TFL', 'QF', 'GC', 'TA']):
        for col, side in enumerate(['L', 'R']):
            ax = axes[row, col]
            data = df_var[['BBS', f'{side}_{sensor}']].reset_index(drop=True)
            x = data[f'{side}_{sensor}'].astype(float)
            y = data['BBS'].astype(float)
            r, p = pearsonr(x, y)
            sns.regplot(ax=ax, ci=0, x=x, y=y)

            ax.set_ylabel('BBS_variation', fontsize=15)
            ax.set_xlabel(f'RMS_variation', fontsize=15)
            ax.set_title(f'{side}_{sensor}\nr={r: 0.2f}, p={p: 0.2f}', fontsize=15)

    fig.suptitle(f'{mv}', fontweight="bold", fontsize=15)


def pvalue_cut(pvalue):
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    else:
        return ""


def icc_df(indiv_params, cond):
    df = pd.DataFrame()
    for param in indiv_params:
        info = []
        motion = param.info['motion']
        if cond == 'group':
            info += [param.info['group'] == 'Disabled']
        else:
            info += [param.info['BBS'] < 45]
        info += [param.info['sid']]
        info += [motion.split('_')[-1]]
        info += ['_'.join(motion.split('_')[:2])]
        rms = list(np.nanmax(param.rms.data, axis=0))

        new_df = pd.DataFrame(info + rms).T
        new_df.columns = ['Group', 'Sid', 'Power', 'Motion'] + COLUMNS
        df = pd.concat([df, new_df])
    return df


def rms_icc(df, group, power, sensor):
    cond_df = df[
        (df['Group'] == group) & (df['Power'] == power)].reset_index(drop=True)
    # for sid in cond_df['Sid'].unique():
    #     if len(cond_df[cond_df['Sid'] == sid]) < 5:
    #         cond_df = cond_df.loc[cond_df['Sid'] != sid].reset_index(drop=True)
    icc = pg.intraclass_corr(data=cond_df,
                             targets='Motion',
                             raters='Sid', ratings=sensor, nan_policy='omit')
    return icc
