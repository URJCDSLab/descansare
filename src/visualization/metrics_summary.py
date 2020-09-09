import itertools
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, KFold
from tabulate import tabulate
from src.visualization import plot_utils

warnings.filterwarnings("ignore")


def mae_std(y_true, y_pred):
    return np.std(np.abs(y_true - y_pred))


def res_std(y_true, y_pred):
    return np.std(y_true - y_pred)


def mae(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)


def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))


def rmse_std(y_true, y_pred):
    return np.std(np.sqrt((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def medape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100


def per_95ape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.percentile(np.abs((y_true - y_pred) / y_true), 95) * 100


def per_5ape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.percentile(np.abs((y_true - y_pred) / y_true), 5) * 100


# def year_group(y, y_preds):
#     for year in y.index.year.unique():
#         yield year, (y[y.index.year == year], y_preds[y_preds.index.year == year])


def acc(y, y_preds, acc_margin=0.5):
    return sum(np.abs(y - y_preds) < acc_margin) / len(y)


def acc_1m(y, y_preds, acc_margin=1):
    return sum(np.abs(y - y_preds) < acc_margin) / len(y)


def interval_group(y, y_preds):
    floor = int(np.floor(min(y)))
    ceil = int(np.ceil(max(y)))

    groups = list(zip(range(floor, ceil, 10), range(floor + 10, ceil + 1, 10)))

    for l, h in groups:
        mask = (y >= l) & (y < h)
        yield str((l, h)), (y[mask], y_preds[mask])


def metrics_summary(y, y_preds, groupby_f={'int': interval_group},
                    metrics=(rmse, rmse_std, mae, mae_std), metrics_kwargs={}):
    index = []
    data = []

    def _fmt(_group_name, _name):
        return ':'.join([str(x) for x in [_group_name, _name] if x != ''])

    def _add(_y, _y_preds, _group_name, _name):
        index.append(_fmt(_group_name, _name))
        data.append(OrderedDict([('Nº', len(_y))] + [
            (metric.__name__, round(metric(_y, _y_preds), 3)) if len(_y) > 0 else (metric.__name__, '-') for metric in
            metrics]))

    _add(y, y_preds, 'Total', '')

    for group_name, groupby in groupby_f.items():
        for name, (group_y, group_y_preds) in groupby(y, y_preds):
            _add(group_y, group_y_preds, group_name, name)

    summary_df = pd.DataFrame(data, index=index)
    summary_df = summary_df.round(3)
    return summary_df


def err_hist(y_pred, y):
    error_test = pd.DataFrame(y - y_pred)
    print("\n\n")
    error_test.hist()
    plt.title('Histograma del error (esperado-predicción)')
    plt.show()


def pred_hist(y_pred, y):
    bins = np.linspace(min(y.min(), y_pred.min()), max(y.max(), y_pred.max()), 15)

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax1.set_title('Histograma de las predicciones')
    ax1.hist(y_pred, bins=bins)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Histograma de y')
    ax2.hist(y, bins=bins)
    plt.show()


def plot_corr_matrix(y_pred, y):
    print("\n\n\n")
    print('\033[1m' + "SQR - Predicción SQR" + '\033[0m')

    plot_utils.plot_corr_matrix(df=pd.DataFrame({'test': y, 'test_p': y_pred}))
    plt.show()


def plot_residuo(y_pred, y):
    print('\033[1m' + "Residuos" + '\033[0m')
    residuo = y - y_pred

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    sns.regplot(y, residuo, ax=ax1)
    ax1.set_title('Scatter y-residuo')
    ax1.set_xlabel('y')
    ax1.set_ylabel('residuo')

    ax2 = fig.add_subplot(122)
    ax2.set_title('Scatter y_pred-residuo')
    sns.regplot(y_pred, residuo, ax=ax2)
    ax2.set_xlabel('y_pred')
    ax2.set_ylabel('residuo')
    plt.show()


def temporal_reg(model, x, y, split_fn=lambda index: index.year < 2019,
                 groupby_f={'int': interval_group},
                 metrics=(rmse, rmse_std, mae, mae_std), metrics_kwargs={}, plots=(), error=False,
                 store_path=None):
    mask = split_fn(x.index)

    x_train = x[mask]
    y_train = y[mask]

    x_test = x[~mask]
    y_test = y[~mask]

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    temp_plot_params = {'error': error}

    return reg_summary(y_pred, y_test, groupby_f=groupby_f, metrics=metrics, metrics_kwargs=metrics_kwargs, plots=plots,
                       store_path=store_path)


def cross_val_reg(model, x, y, cv=10, transformer=None, groupby_f={'int': interval_group},
                  metrics=(rmse, rmse_std, mae, mae_std), metrics_kwargs={}, plots=(),
                  prediction_interval_kwargs=None, error=False, store_path=None):
    if transformer is not None:
        transformer = transformer.fit(np.array(y).reshape(-1, 1))
        y_norm = transformer.transform(np.array(y).reshape(-1, 1))
        y_pred = cross_val_predict(model, x, y_norm, cv=cv)
        y_pred = transformer.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    else:
        y_pred = cross_val_predict(model, x, y, cv=cv)
        y_pred = y_pred.reshape(-1, )

    y_pred = pd.Series(y_pred, index=y.index)

    temp_plot_params = {'error': error}

    if prediction_interval_kwargs is not None:
        temp_plot_params['plot_bands'] = prediction_interval(model, x, y, **prediction_interval_kwargs)

    return reg_summary(y_pred, y, groupby_f=groupby_f, metrics=metrics, metrics_kwargs=metrics_kwargs, plots=plots,
                       store_path=store_path)


def prediction_interval(model, x, y, alpha):
    model.set_params(loss='quantile')
    model.set_params(alpha=alpha)

    y_upper = cross_val_predict(model, x, y)

    model.set_params(alpha=1 - alpha)

    y_lower = cross_val_predict(model, x, y)

    return pd.DataFrame({'upper': y_upper, 'lower': y_lower, 'timestamp': y.index})


def reg_summary(y_pred, y, groupby_f={'int': interval_group},
                metrics=(rmse, rmse_std, mae, mae_std), metrics_kwargs={}, plots=(),
                layout='fancy_grid',
                store_path=None):
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y.index)

    if store_path is not None:
        pd.DataFrame({'vacio_pred': y_pred.values, 'vacio_real': y.values}, index=y.index).to_parquet(
            store_path)

    metrics = metrics_summary(y, y_pred, groupby_f, metrics, metrics_kwargs)
    print('\033[1m' + "Metrics summary: " + '\033[0m')
    print(tabulate(metrics, headers='keys', tablefmt=layout))

    error_test = pd.DataFrame(y - y_pred).describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T.\
        drop(columns=['mean', 'count', 'std'])
    error_test[' '] = 'Error'
    error_test.set_index(' ', inplace=True)
    print("\n")
    print('\033[1m' + "Resid statistics: real-predicted" + '\033[0m')
    print(tabulate(error_test.round(3), headers='keys', tablefmt=layout))

    print("\n\n")

    for plot in plots:
        plot(y_pred, y)
        print("\n\n")

    result = pd.DataFrame((y - y_pred).values, columns=['pred_error'], index=y.index)
    result['y'] = y
    result['y_pred'] = y_pred
    return result


def print_variables_performance(model, cv, df, y, topics):
    topic_variables = {}

    for topic in topics:
        topic_variables[topic] = [col for col in df.columns if col.find(topic) != -1]

    topic_subsets = []
    for mask in list(itertools.product([1, 0], repeat=len(topics))):
        subset = [col for include, col in zip(mask, topics) if include]
        if len(subset) > 0:
            topic_subsets.append(subset)

    topic_subsets = sorted(topic_subsets, key=len)

    y_pred = cross_val_predict(model, df, y, cv=cv)
    y_pred = y_pred.reshape(-1, )
    y_pred = pd.Series(y_pred, index=y.index)

    completo = metrics_summary(y, y_pred, groupby_f={})
    completo.index = [' ']

    metrics_quality = [completo]

    for topic_subset in topic_subsets:
        subset_varibles = itertools.chain.from_iterable([topic_variables[topic] for topic in topic_subset])

        y_pred = cross_val_predict(model, df.drop(columns=subset_varibles), y, cv=cv)
        y_pred = y_pred.reshape(-1, )
        y_pred = pd.Series(y_pred, index=y.index)

        local_metrics = metrics_summary(y, y_pred, groupby_f={})
        local_metrics.index = [", ".join(topic_subset)]

        metrics_quality.append(local_metrics)

    metrics_quality = pd.concat(metrics_quality)

    metrics_quality.index.name = 'Variables Excluidas'

    metrics_quality.drop(columns='Nº', inplace=True)

    return print(tabulate(metrics_quality, headers='keys', tablefmt='fancy_grid'))


def gam_cross_val_reg(model, x, y, cv=10, groupby_f={'int': interval_group},
                      metrics=(rmse, rmse_std, mae, mae_std), metrics_kwargs={}, plots=(),
                      store_path=None, gam_kwargs={"progress": False}, weights=None):
    kf = KFold(n_splits=cv)
    predictions = []

    for train_index, val_index in kf.split(x):

        # machine learning model sketch
        model_k = model

        # training set
        time_train_index = x.iloc[train_index].index
        X_train = x.loc[time_train_index]
        y_train = y.loc[time_train_index]

        # validation set
        time_val_index = x.iloc[val_index].index
        X_val = x.loc[time_val_index]

        if weights is None:
            # train model
            model_k.gridsearch(X_train.values, y_train.values, **gam_kwargs)

        else:
            row_weights = weights.loc[time_train_index].values

            model_k.gridsearch(X_train.values, y_train.values, weights=row_weights, **gam_kwargs)

        # predictions
        y_pred = model_k.predict(X_val)

        # store predictions and order
        predictions.extend(zip(y_pred, X_val.index))

    predictions.sort(key=lambda x: x[1])
    predictions = list(zip(*predictions))[0]

    # print report
    return predictions, reg_summary(predictions, y, groupby_f=groupby_f, metrics=metrics, metrics_kwargs=metrics_kwargs,
                                    plots=plots, store_path=store_path)


def gam_temporal_reg(model, x_train, x_test, y_train, y_test,
                 groupby_f={'int': interval_group},
                 metrics=(rmse, rmse_std, mae, mae_std), metrics_kwargs={}, plots=(), error=False,
                 store_path=None, gam_kwargs={"progress": False}):

    model.gridsearch(x_train.values, y_train.values, **gam_kwargs)

    y_pred = model.predict(x_test)

    temp_plot_params = {'error': error}

    return y_pred, reg_summary(y_pred, y_test, groupby_f=groupby_f, metrics=metrics, metrics_kwargs=metrics_kwargs,
                               plots=plots, store_path=store_path)