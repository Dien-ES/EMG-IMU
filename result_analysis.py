import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

sns.set_style("whitegrid")


def comb_comparison_result(result_df, x, order, title, figsize=(21,15)):
    result_df_melt = result_df.melt(id_vars=['Sensors', 'Parameters'],
                                    value_vars=['AUC', 'Accuracy',
                                                'Sensitivity', 'F-score',
                                                'MCC'],
                                    var_name='Metric', value_name='Score')

    plt.figure(figsize=figsize)
    sns.pointplot(data=result_df_melt,
                  x=x,
                  y='Score',
                  hue='Metric',
                  order=result_df.pivot_table(index=x,
                                              values=order,
                                              aggfunc='mean').
                  sort_values(by=order, ascending=False).index)

    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='lower left')
    plt.ylabel('Score', fontsize=25)
    plt.xlabel(x, fontsize=25)
    plt.xticks(rotation=90)
    plt.title(title, size=25)


def params_comb_filter(result_df, params):
    max_comb = []
    for i in range(1, len(params)+1):
        max_comb += list(combinations(params, i))
    max_comb = [str(i) for i in max_comb]
    bool_idx = [i in max_comb for i in result_df['Parameters']]
    return result_df.loc[bool_idx, :].reset_index(drop=True)


def oversampling_comparison_result(params, x, metric, title, figsize=(21, 15)):
    df = pd.read_csv('../parameter/onset/test_results_(EMG+IMU)_(oversampling)_(2fold).csv')
    df = params_comb_filter(df, params)
    df['is_oversampling'] = True

    df2 = pd.read_csv('../parameter/onset/test_results_(EMG+IMU)_(not oversampling)_(2fold).csv')
    df2 = params_comb_filter(df2, params)
    df2['is_oversampling'] = False

    df3 = pd.concat([df, df2], axis=0).reset_index(drop=True)

    df4 = df3.melt(id_vars=[x, 'is_oversampling'],
                   value_vars=[metric],
                   var_name='Metric', value_name=metric)

    plt.figure(figsize=figsize)
    sns.barplot(data=df4, x=x, y=metric,
                hue='is_oversampling', palette='Set2',
                order=df.pivot_table(index=x, values=metric, aggfunc='mean').
                sort_values(by=metric, ascending=False).index)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, loc='lower left')
    plt.ylabel(metric, fontsize=25)
    plt.xlabel('', fontsize=25)
    plt.xticks(rotation=90)
    plt.title(title, size=25)

