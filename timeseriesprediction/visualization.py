import os
import random
import pathlib
import numpy as np
from darts import TimeSeries
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd

plt.ioff()


def apply_anomalies(ds, anomaly_dates):
    is_in_indices = ds.time_index[ds.time_index.floor('D').isin(anomaly_dates)]

    new_vals = []
    anomal_count = 0
    for i in ds:
        if i.time_index.isin(is_in_indices):
            coef = 1.9 + (random.random() / 5) if int(
                str(i.time_index.strftime("%Y-%m-%d")[0])[-1]) % 2 == 0 else 0.25 + (random.random() / 10)
            new_vals.append(i.values() * coef)
            anomal_count += 1
        else:
            new_vals.append(i.values())
    return TimeSeries.from_times_and_values(ds.time_index, new_vals)


def visualize_ts(trainer, show_plot=False, save_plot=False, i_component=-1, y_id=0):
    if 'train' in trainer.ts_ds.mode:
        preds_train = trainer.evaluate_train['preds'].copy()
        # if len(trainer.evaluate_train['ys'].shape) == 2:
        #     pass
        #     # ys_train = trainer.evaluate_train['ys'][:, i_component]
        #     # preds_train = preds_train[:, i_component]
        #
        #     # pred_values_train = TimeSeries.from_times_and_values(
        #     #     pd.DatetimeIndex(trainer.data_train[trainer.params['seq_len_in']:]['ds'].values,
        #     #                      freq=trainer.params['time_freq']), preds_train)
        # else:
        #     pass
        #     # pred_values_train = TimeSeries.from_times_and_values(
        #     #     pd.DatetimeIndex(trainer.data_train[
        #     #                      i_component + trainer.params['seq_len_in']:i_component - trainer.params[
        #     #                          'seq_len_out'] + 1][
        #     #                          'ds'].values,
        #     #                      freq=trainer.params['time_freq']),
        #     #     preds_train)

        actual_values_train = trainer.data_train
        if trainer.params['show_anomalies_in_plot'] and False:
            actual_values_train_anomal = apply_anomalies(actual_values_train, trainer.ts_ds.anomaly_dates)

    if 'test' in trainer.ts_ds.mode:

        preds_test = trainer.evaluate_test['preds'].copy()
        num_components = trainer.evaluate_test['preds'].shape[1]
        ys_test = trainer.evaluate_test['ys'].copy()
        pred_values_test = TimeSeries.from_times_and_values(
            pd.DatetimeIndex(trainer.data_val[trainer.params['seq_len_in']+i_component:i_component-trainer.params['seq_len_out']+1]['ds'].values,
                             freq=trainer.params['time_freq']),
                            preds_test[:, i_component, y_id])
        # trainer.data_val[trainer.params['seq_len_in']+i_component:i_component - trainer.params[
        #                                      'seq_len_out'] + 1]['ds'].values
        actual_values_test = trainer.data_val[trainer.params['seq_len_in']:]

        if trainer.params['show_anomalies_in_plot'] and False:
            actual_values_test_anomal = apply_anomalies(actual_values_test, trainer.ts_ds.anomaly_dates)

    if os.path.isfile(os.path.join(trainer.params['plots_directory'],
                               'i_' + str(i_component) + "_" + str(trainer.params['target_columns'][y_id]) + '.json')):

        fig = plotly.io.read_json(os.path.join(trainer.params['plots_directory'],
                               'i_' + str(i_component) + "_" + str(trainer.params['target_columns'][y_id]) + '.json'))

        if fig:
            if 'test' in trainer.ts_ds.mode:
                fig.add_trace(
                    go.Scatter(x=pred_values_test.time_index, y=pred_values_test.values().squeeze(),
                               name=trainer.params['model']))

    else:
        fig = go.Figure()
        # scatter1 = go.Scatter(x=actual_values_train.time_index, y=actual_values_train.values().squeeze(),
        # name='actual train normal', line=dict(color='mediumblue'))

        # fig.add_trace(scatter1)

        # fig.add_trace(go.Scatter(x=pred_values_train.time_index, y=pred_values_train.values().squeeze(),
        # name='prediction train'))

        # if trainer.params['show_anomalies_in_plot']:
        #     if 'test' in trainer.ts_ds.mode:
        #         fig.add_trace(go.Scatter(x=actual_values_test_anomal.time_index, y=0.2 * np.abs(
        #             actual_values_test_anomal.values().squeeze() - pred_values_test.values().squeeze()),
        #                                  name='anomaly detector', ))
        #         fig.add_trace(
        #             go.Scatter(x=actual_values_test_anomal.time_index, y=actual_values_test_anomal.values().squeeze(),
        #                        name='actual test with anomaly'))

        # fig.add_trace(go.Scatter(x=actual_values_test.time_index, y=actual_values_test.values().squeeze(), name='actual test normal',
        #                          line=dict(color='mediumblue')))
        if trainer.ts_ds.mode == 'train':
            fig.add_trace(go.Scatter(
                x=trainer.data_train['ds'],
                y=trainer.data_train['y_' + str(y_id)],
                name='ground truth'))
        elif trainer.ts_ds.mode == 'test':
            fig.add_trace(go.Scatter(
                x=trainer.data_val['ds'],
                y=trainer.data_val['y_' + str(y_id)],
                name='ground truth'))
        elif trainer.ts_ds.mode == 'train_test':
            fig.add_trace(go.Scatter(
                # [trainer.params['seq_len_in']:-num_components + 1]
                x=pd.concat([trainer.data_train, trainer.data_val[trainer.params['seq_len_in']:]])['ds'].values,
                y=pd.concat([trainer.data_train, trainer.data_val[trainer.params['seq_len_in']:]])['y_' + str(y_id)].values,
                name='ground truth'))


        if 'test' in trainer.ts_ds.mode:
            fig.add_trace(
                go.Scatter(x=pred_values_test.time_index, y=pred_values_test.values().squeeze(), name=trainer.params['model']))


    fig.update_layout(
        title=str(trainer.params['target_columns'][y_id]) + ' --- ' + str(trainer.params['time_freq']) + '---' +' Timestep:'+ str(i_component+1),
        xaxis_title="Date Time",
        yaxis_title="Value"
    )

    # Save the figure to a JSON file
    fig.write_json(os.path.join(trainer.params['plots_directory'],
                                'i_' + str(i_component) + "_" + str(trainer.params['target_columns'][y_id]) + '.json'))


    if save_plot:
        path = pathlib.Path(trainer.params['plots_directory'])
        path.mkdir(parents=True, exist_ok=True)
        fig.write_image(os.path.join(trainer.params['plots_directory'],
                                     'i_' + str(i_component) + "_" + str(trainer.params['target_columns'][y_id]) + '_' +
                                     trainer.params['ts_plot_filename']), scale=10, width=1200)

    if show_plot:
        fig.show()


    return True


def visualize_metrics(trainer, show_plot=False, save_plot=False):
    for metric in ['rmse', 'mae', 'smape', 'loss']:
        title = str(trainer.params['time_freq']) + ' | ' + metric

        train_metric = trainer.metric_tracker.train_loss[metric]
        #     train_losses = np.array(cb.train_loss)[np.array(cb.train_loss)<100_000]
        val_metric = trainer.metric_tracker.val_loss[metric]
        #     val_losses = np.array(cb.val_loss)[np.array(cb.train_loss)<100_000]
        x = range(len(train_metric))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(x, train_metric, s=10, c='b', marker="s", label='train_' + metric)
        ax1.scatter(x, val_metric, s=10, c='r', marker="o", label='test_' + metric)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.title(title)

        if save_plot:
            plt.savefig(os.path.join(trainer.params['plots_directory'], metric + '.png'), dpi=100.0)
        if show_plot:
            plt.show()

    return True
