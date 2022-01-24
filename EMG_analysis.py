import pickle
from statannot import add_stat_annotation
import pingouin as pg

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
        df = df.loc[df['is_secondDay']].reset_index(drop=True)
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
                        hue='Group', palette='Set3',
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
                        palette='Set3',
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
