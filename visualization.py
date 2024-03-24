import os
import pathlib
from darts import TimeSeries
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd

plt.ioff()

def visualize_ts(trainer, show_plot=False, save_plot=False, i_component=-1, y_id=0):
    if 'train' in trainer.ts_ds.mode:
        preds_train = trainer.evaluate_train['preds'].copy()

        actual_values_train = trainer.data_train
    if 'test' in trainer.ts_ds.mode:

        preds_test = trainer.evaluate_test['preds'].copy()
        num_components = trainer.evaluate_test['preds'].shape[1]
        pred_values_test = TimeSeries.from_times_and_values(
            pd.DatetimeIndex(trainer.data_val[trainer.params['seq_len_in']:-num_components + 1]['ds'].values,
                             freq=trainer.params['time_freq']),

            preds_test[:, i_component, y_id])
        actual_values_test = trainer.data_val[trainer.params['seq_len_in']:]

    fig = go.Figure()

    if trainer.ts_ds.mode == 'train':
        fig.add_trace(go.Scatter(
            x=trainer.data_train['ds'],
            y=trainer.data_train['y_' + str(y_id)],
            name='actual normal'))
    elif trainer.ts_ds.mode == 'test':
        fig.add_trace(go.Scatter(
            x=trainer.data_val['ds'],
            y=trainer.data_val['y_' + str(y_id)],
            name='actual normal'))
    elif trainer.ts_ds.mode == 'train_test':
        fig.add_trace(go.Scatter(
            x=pd.concat([trainer.data_train, trainer.data_val[trainer.params['seq_len_in']:-num_components + 1]])['ds'].values,
            y=pd.concat([trainer.data_train, trainer.data_val[trainer.params['seq_len_in']:-num_components + 1]])['y_' + str(y_id)].values,
            name='actual normal'))

    if 'test' in trainer.ts_ds.mode:
        fig.add_trace(
            go.Scatter(x=pred_values_test.time_index, y=pred_values_test.values().squeeze(), name='prediction test'))

    fig.update_layout(
        title=str(trainer.params['target_columns'][y_id]) + ' --- ' + str(trainer.params['time_freq']),
        xaxis_title="Date Time",
        yaxis_title="Value"
    )

    if save_plot:
        path = pathlib.Path(trainer.params['plots_directory'])
        path.mkdir(parents=True, exist_ok=True)
        fig.write_image(os.path.join(trainer.params['plots_directory'],
                                     'i_' + str(i_component) + "_" + str(trainer.params['target_columns'][y_id]) + '_' +
                                     trainer.params['ts_plot_filename']), scale=10,width=800)
    if show_plot:
        fig.show()
    return True

def visualize_metrics(trainer, save_plot=True):
    for metric in ['rmse', 'mae', 'smape', 'loss']:
        title = str(trainer.params['time_freq']) + ' | ' + metric
        train_metric = trainer.metric_tracker.train_loss[metric]
        val_metric = trainer.metric_tracker.val_loss[metric]
        x = range(len(train_metric))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(x, train_metric, s=10, c='b', marker="s", label='train_' + metric)
        ax1.scatter(x, val_metric, s=10, c='r', marker="o", label='test_' + metric)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.title(title)
        if save_plot:
            plt.savefig(os.path.join(trainer.params['plots_directory'], metric + '.png'), dpi=100.0)
        plt.show()

    return True