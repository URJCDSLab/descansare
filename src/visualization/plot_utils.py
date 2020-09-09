import collections
import itertools
import math
import warnings
from textwrap import wrap

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from highcharts import Highstock
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
# from modeling_utilities.utils import maybe_map_titles

warnings.filterwarnings("ignore")

def scatter_wy_nearly_pc(df, y_name='ind_vacio_de_cam', cmap='seismic', columns=4, figsize=(15, 15)):
    corr = df.corr() * np.triu(np.ones(len(df.columns)))
    corr_pairs = corr.unstack()
    corr_pairs = corr_pairs[(corr_pairs.abs() > 0.9) & (corr_pairs.abs() < 0.9999)]
    pairs = corr_pairs.to_dict().keys()

    rows = math.ceil(len(pairs) / columns)

    fig = plt.figure(figsize=figsize)

    for index, (col1, col2) in enumerate(pairs):
        ax = fig.add_subplot(rows, columns, index + 1)
        scatter_wy(df[col1], df[col2], y_name=y_name, cmap=cmap, ax=ax)

    fig.tight_layout()


def scatter_wy_group(df, y, cmap='seismic', plt_num_columns=4, figsize=(5, 5)):
    fig = plt.figure(figsize=figsize)
    pairs = list(itertools.combinations(df.columns, 2))
    filas = math.ceil(len(pairs) / plt_num_columns)

    for index, (c1, c2) in enumerate(pairs):
        ax = fig.add_subplot(filas, plt_num_columns, index + 1)
        scatter_wy(df[c1], df[c2], y, ax=ax)
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)

    fig.tight_layout()


def scatter_wy(column_x, column_y, y, cmap='seismic', figsize=(5, 5), ax=None):
    y_gradient = QuantileTransformer(output_distribution='uniform').fit_transform(y.reshape(-1, 1)).reshape(-1)
    cmap = plt.get_cmap(cmap)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    ax.scatter(column_x, column_y, c=np.array([cmap(gradient) for gradient in y_gradient]))


def corrfunc(x, y, **kws):
    (r, p) = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate('r = {:.2f} '.format(r) + p_stars, xy=(0.05, 0.9),
                xycoords=ax.transAxes)


def annotate_colname(x, **kws):
    ax = plt.gca()
    ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
                fontweight='bold')


def plot_corr_matrix(df, size=2.5):
    """
      source: https://stackoverflow.com/a/48145509
  """

    g = sns.PairGrid(df, palette=['red'], size=size)

    # Use normal regplot as `lowess=True` doesn't provide CIs.

    g.map_upper(sns.regplot, scatter_kws={'s': 10})
    g.map_diag(sns.distplot)
    # g.map_diag(annotate_colname)
    g.map_lower(sns.kdeplot, cmap='Blues_d')
    g.map_lower(corrfunc)

    # Remove axis labels, as they're in the diagonals.

    for ax in g.axes.flatten():
        ax.set_ylabel('')
        ax.set_xlabel('')
    return g


def temp_plot(*args, shift=0, plot_type='regplot', **kws):
    x = args[0]
    y = np.roll(np.array(args[-1]), -shift)
    y[-1] = 0

    if plot_type == 'reg':
        fill_with = sns.regplot
    elif plot_type == 'kde':
        fill_with = sns.kdeplot
    else:
        raise ValueError('unkwnon plot type {}'.format(kws['plot_type']))

    fill_with(x, y, **kws)
    corrfunc(x, y, **kws)


def temp_plot_corr_matrix(df, shift=1, plot_type='regplot'):
    g = sns.PairGrid(df, palette=['red'])

    g.map_upper(temp_plot, shift=shift, plot_type=plot_type)
    g.map_lower(temp_plot, shift=shift, plot_type=plot_type)
    g.map_diag(temp_plot, shift=shift, plot_type=plot_type)


def scatter_with(data, columnas, columna_y, plt_num_columns=3, figsize=(15, 10)):
    fig = plt.figure(figsize=figsize)
    filas = math.ceil(len(columnas) / plt_num_columns)

    for index, columna in enumerate(columnas):
        ax = fig.add_subplot(filas, plt_num_columns, index + 1)
        sns.regplot(data[columna], data[columna_y], ax=ax)
        ax.set_xlabel(columna)
        ax.set_ylabel(columna_y)

    fig.tight_layout()


def hist_plot(df, figsize=(4, 4), columns=1, include_mean=True, quantiles=[0.25, 0.5, 0.75], legend_loc=0,
              group_by=None, bins=10, log=False, titles_map=None):
    fig = plt.figure(figsize=figsize)

    rows = math.ceil(len(df.columns) / columns)
    legend_set = False

    for index, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, columns, index + 1)

        if group_by is not None:
            names, groups = zip(*df[column].groupby(group_by))
            plt.hist(list(groups), label=list(names), bins=bins, log=log)
            legend_set = True
        else:
            df[column].hist(ax=ax, bins=bins, log=log)

        ax.set_title("\n".join(wrap(maybe_map_titles(column, titles_map), 40)))

        if quantiles:
            for i, quantil in enumerate(quantiles):
                try:
                    q = df[column].quantile(quantil)
                    ax.axvline(q, color=cm.PiYG(i + 1), label='{}%'.format(quantil))
                    legend_set = True
                except Exception as e:
                    print("Err calculating quantile {} in {}. {}".format(quantil, column, e))

        if legend_set:
            ax.legend(loc=legend_loc)

    fig.tight_layout()


def rotate_axis(labels, ha='right', rotation_mode='anchor'):
    for label in labels:
        label.set_horizontalalignment(ha)
        label.set_rotation(45)
        label.set_rotation_mode(rotation_mode)


def heatmap_corrmap(df, figsize=(15, 15), include_corr_num=True, sort_by_first_corr=True, sort_ascending=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    corr = df.corr()

    if sort_by_first_corr:
        first_col = corr[corr.columns[0]].sort_values(ascending=sort_ascending)
        corr = df[first_col.index].corr()

    labels = corr.where(np.triu(np.ones(corr.shape)).astype(np.bool))
    labels = labels.round(2)
    labels = labels.replace(np.nan, ' ', regex=True)

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    g = sns.heatmap(corr, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, cmap='RdYlGn_r', mask=mask, annot=True)

    rotate_axis(ax.get_xticklabels())


def heatmap_corrmap_v2(df1, df2, figsize=(15, 15), include_corr_num=True, sort_ascending=False):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    corr = pd.DataFrame(df1.corrwith(df2[df2.columns[0]]), columns=[df2.columns[0]])

    for i in df2.columns[1:]:
        df1.corrwith(df2[i])
        corr[i] = df1.corrwith(df2[i])

    g = sns.heatmap(corr, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, cmap='RdYlGn_r',  # mask=mask,
                    annot=True)


def temporal_plot(df, title="", plot_bands=None):
    y = df['y']
    y_pred = df['y_pred']

    result = df.copy()
    result['error_abs'] = (y - y_pred).abs()

    H3 = Highstock(width=970, height=600)


    H3.add_data_set(result[['e_inicio_carga', 'error_abs']].values.tolist(), 'column', 'Error absoluto', yAxis=1,
                    color='#808080',
                    tooltip={
                        'valueDecimals': 2
                    }
                    )

    H3.add_data_set(result[['e_inicio_carga', 'y_pred']].values.tolist(), 'line', name='Valor estimado',
                    color='#C0043D',
                    lineWidth=1,
                    tooltip={
                        'valueDecimals': 2
                    },
                    marker={
                        'symbol': 'circle',
                        'states': {
                            'hover': {
                                'enabled': True,
                                'fillColor': '#B22222',
                                'lineColor': '#FFFFFF',
                                'lineWidth': 3
                            }
                        }
                    }
                    )

    H3.add_data_set(result[['e_inicio_carga', 'y']].values.tolist(), 'line', 'Valor real',
                    color='#0058B0',
                    lineWidth=1,
                    tooltip={
                        'valueDecimals': 2
                    },
                    marker={
                        'symbol': 'circle',
                        'states': {
                            'hover': {
                                'enabled': True,
                                'fillColor': '#191970',
                                'lineColor': '#FFFFFF',
                                'lineWidth': 3
                            }
                        }
                    }
                    )

    if plot_bands is not None:
        H3.add_data_set(plot_bands[['timestamp', 'lower', 'upper']].values.tolist(), 'arearange', name='Prediction interval',
                        # color='#C0043D',
                        # lineWidth=1,
                        # tooltip={
                        #     'valueDecimals': 2
                        # },
                        # marker={
                        #     'symbol': 'circle',
                        #     'states': {
                        #         'hover': {
                        #             'enabled': True,
                        #             'fillColor': '#B22222',
                        #             'lineColor': '#FFFFFF',
                        #             'lineWidth': 3
                        #         }
                        #     }
                        # }
                        )

    options = {

        'rangeSelector': {
            'selected': 1
        },
        'title': {
            'text': title
        },

        'xAxis': {
            'title': {
                'text': 'Ciclos'
            }
        },
        'yAxis': [{

            'labels': {
                'align': 'right',
                'x': -3
            },
            'title': {
                'text': 'Vacío de cámara'
            },
            'height': '60%',
            'lineWidth': 2,
            'plotBands': {
                'color': '#F0F0F0',
                'from': result['y'].mean() - result['y'].std(),
                'to': result['y'].mean() + result['y'].std()
            },
            #                         "min":5,
            #                         "max": 12
        },
            {
                'labels': {
                    'align': 'right',
                    'x': -3
                },
                'title': {
                    'text': 'Error de predicción'
                },
                'top': '75%',
                'height': '25%',
                'offset': 0,
                'lineWidth': 2,
                'plotBands': {
                    'color': '#F0F0F0',
                    'from': 0,
                    'to': 0.5
                }
            }],
        'rangeSelector': {
            'selected': 2
        }
    }

    H3.set_dict_options(options)
    return H3


def temporal_plot_v2(df, title="", plot_bands=None):
    y = df['y']
    y_pred = df['y_pred']

    result = df.copy()

    H3 = Highstock(width=970, height=600)


    H3.add_data_set(result[['e_inicio_carga', 'y_pred']].values.tolist(), 'line', name='Valor estimado',
                    color='#C0043D',
                    lineWidth=1,
                    tooltip={
                        'valueDecimals': 2
                    },
                    marker={
                        'symbol': 'circle',
                        'states': {
                            'hover': {
                                'enabled': True,
                                'fillColor': '#B22222',
                                'lineColor': '#FFFFFF',
                                'lineWidth': 3
                            }
                        }
                    }
                    )

    H3.add_data_set(result[['e_inicio_carga', 'y']].values.tolist(), 'line', 'Valor real',
                    color='#0058B0',
                    lineWidth=1,
                    tooltip={
                        'valueDecimals': 2
                    },
                    marker={
                        'symbol': 'circle',
                        'states': {
                            'hover': {
                                'enabled': True,
                                'fillColor': '#191970',
                                'lineColor': '#FFFFFF',
                                'lineWidth': 3
                            }
                        }
                    }
                    )


    if plot_bands is not None:
        H3.add_data_set(plot_bands[['timestamp', 'lower', 'upper']].values.tolist(), 'arearange',
                        name='Prediction interval',
                        # color='#C0043D',
                        # lineWidth=1,
                        # tooltip={
                        #     'valueDecimals': 2
                        # },
                        # marker={
                        #     'symbol': 'circle',
                        #     'states': {
                        #         'hover': {
                        #             'enabled': True,
                        #             'fillColor': '#B22222',
                        #             'lineColor': '#FFFFFF',
                        #             'lineWidth': 3
                        #         }
                        #     }
                        # }
                        )

    options = {

        'rangeSelector': {
            'selected': 1
        },
        'title': {
            'text': title
        },

        'xAxis': {
            'title': {
                'text': 'Ciclos'
            }
        },
        'yAxis': [{

            'labels': {
                'align': 'right',
                'x': -3
            },
            'title': {
                'text': 'Vacío de cámara'
            },
            'lineWidth': 2,
            'plotBands': {
                'color': '#F0F0F0',
                'from': result['y'].mean() - result['y'].std(),
                'to': result['y'].mean() + result['y'].std()
            },

        }],
        'rangeSelector': {
            'selected': 2
        }
    }

    H3.set_dict_options(options)
    return H3


palette = ['#845ec2', '#d65db1', '#ff6f91', '#ff9671', '#ffc75f', '#f9f871', '#008f7a', '#c34a36', '#926c00'][::-1]


def multiple_temporal_plot(df, columns, y_name, title="", feature_range=None, width=970, height=600):
    y = df[y_name]
    result = df.copy()

    if feature_range is not None:
        result[columns] = pd.DataFrame(MinMaxScaler(feature_range=(0, 5)).fit_transform(result[columns]),
                                       index=result.index, columns=columns)

    H3 = Highstock(width=width, height=height)

    for i, column in enumerate(columns):
        color = palette[i]

        H3.add_data_set(result[['e_inicio_carga', column]].values.tolist(), 'line', name=column,
                        color=color,
                        lineWidth=1,
                        yAxis=0,
                        tooltip={
                            'valueDecimals': 2
                        },
                        marker={
                            'symbol': 'circle',
                            'states': {
                                'hover': {
                                    'enabled': True,
                                    'fillColor': color,
                                    'lineColor': '#FFFFFF',
                                    'lineWidth': 3
                                }
                            }
                        }
                        )

    H3.add_data_set(result[['e_inicio_carga', y_name]].values.tolist(), 'line', y_name,
                    color='#0058B0',
                    lineWidth=1,
                    yAxis=1,
                    tooltip={
                        'valueDecimals': 2
                    },
                    marker={
                        'symbol': 'circle',
                        'states': {
                            'hover': {
                                'enabled': True,
                                'fillColor': '#191970',
                                'lineColor': '#FFFFFF',
                                'lineWidth': 3
                            }
                        }
                    }
                    )

    options = {

        'rangeSelector': {
            'selected': 1
        },
        'title': {
            'text': title
        },

        'xAxis': {
            'title': {
                'text': 'Ciclos'
            }
        },
        'yAxis': [{

            'labels': {
                'align': 'right',
                'x': -3
            },
            'title': {
                'text': 'Variables'
            },
            'height': '60%',
            'lineWidth': 2,
            'plotBands': {
                'color': '#F0F0F0',
                'from': y.mean() - y.std(),
                'to': y.mean() + y.std()
            },
        },
            {
                'labels': {
                    'align': 'right',
                    'x': -3
                },
                'title': {
                    'text': y_name
                },
                'top': '75%',
                'height': '25%',
                'offset': 0,
                'lineWidth': 2,
                'plotBands': {
                    'color': '#F0F0F0',
                    'from': 0,
                    'to': 0.5
                }
            }]
    }

    H3.set_dict_options(options)
    return H3


def var_summary(df, y, figsize=None, friendly_names={}):
    num_filas = len(df.columns)

    if figsize is None:
        figsize = (20, 5 * num_filas)

    fig, big_axes = plt.subplots(figsize=figsize, nrows=num_filas, ncols=1)

    if not isinstance(big_axes, collections.Iterable):
        big_axes = [big_axes]

    for row, big_ax in enumerate(big_axes):
        # https://stackoverflow.com/a/27430940
        big_ax.set_title("%s \n" % friendly_names.get(df.columns[row], df.columns[row]), fontsize=16)
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        big_ax._frameon = False

    for index, fila in enumerate(df.columns):
        ax1 = fig.add_subplot(num_filas, 2, 2 * index + 1)
        ax2 = fig.add_subplot(num_filas, 2, 2 * index + 2)

        sns.regplot(df[fila], y, ax=ax1)
        ax1.set_xlabel('')
        ax2.hist(df[fila])

    fig.tight_layout()
