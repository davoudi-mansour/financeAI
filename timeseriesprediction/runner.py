import sys
from trainer import Trainer
from fine_tuner import FineTuner
from tester import Tester
from grid_searcher import GridSearcher
from visualization import visualize_ts, visualize_metrics
from metrics import mse, rmse, mae, smape
import numpy as np
import torch
from torch import nn


if __name__ == '__main__':
    running_mode = sys.argv[1]
    if running_mode in ('train', 'test', 'finetune', 'gridsearch'):
        print(f'{running_mode} started!')
    else:
        exit(f'running mode "{running_mode}" is not supported!')

    if running_mode == 'train':
        model_trainer = Trainer('./config/train.yml')
        model_trainer.train()
        #-----------------------------------------------------------------------------------------------------#
        print(f'Test Results Normalized: RMSE={model_trainer.evaluate_test["rmse"]} | MAE={model_trainer.evaluate_test["mae"]} | MAPE={model_trainer.evaluate_test["mape"]} | SMAPE={model_trainer.evaluate_test["smape"]} | loss={model_trainer.evaluate_test["loss"]}')

        preds_test_denorm = model_trainer.evaluate_test['preds'].copy()
        ys_test_denorm = model_trainer.evaluate_test['ys'].copy()
        loss_denorm = nn.MSELoss()(torch.tensor(preds_test_denorm),torch.tensor(ys_test_denorm)).item()
        rmse_denorm = rmse(preds_test_denorm, ys_test_denorm, axis=None)
        mae_denorm = mae(preds_test_denorm, ys_test_denorm, axis=None)
        smape_denorm = smape(preds_test_denorm, ys_test_denorm, axis=None)
        print(f'Test Results Denormalized: RMSE={rmse_denorm} | MAE={mae_denorm} | SMAPE={smape_denorm} | loss={loss_denorm}')

        print(f'RMSE Components Normalized : {model_trainer.evaluate_test["rmse_components"]}')
        print(f'MAE Components Normalized : {model_trainer.evaluate_test["mae_components"]}')
        print(f'MAPE Components Normalized : {model_trainer.evaluate_test["mape_components"]}')
        print(f'SMAPE Components Normalized : {model_trainer.evaluate_test["smape_components"]}')
        print(f'loss Components Normalized : {model_trainer.evaluate_test["loss_components"]}')

        # rmse_denorm_components = rmse(preds_test_denorm, ys_test_denorm, axis=0)
        # mae_denorm_components = mae(preds_test_denorm, ys_test_denorm, axis=0)
        # smape_denorm_components = smape(preds_test_denorm, ys_test_denorm, axis=0)
        # loss_denorm_components = mse(preds_test_denorm, ys_test_denorm, axis=0)
        # print(f'RMSE Components Denormalized : {rmse_denorm_components}')
        # print(f'MAE Components Denormalized : {mae_denorm_components}')
        # print(f'SMAPE Components Denormalized : {smape_denorm_components}')
        # print(f'loss Components Denormalized : {loss_denorm_components}')

        #-----------------------------------------------------------------------------------------------------#
        # visualize_ts(model_trainer, show_plot=True, save_plot=False)
        for y_id, target_column in enumerate(model_trainer.params['target_columns']):
            for i_component in range(0, model_trainer.params['seq_len_out']-1):
                visualize_ts(model_trainer, show_plot=False, save_plot=False, i_component=i_component, y_id=y_id)


#         preds_test = model_trainer.evaluate_test['preds'].copy()
#         ys_test = model_trainer.evaluate_test['ys'].copy()
#         loss_run = mse(preds_test, ys_test, axis=None)
#         loss_run2 = nn.MSELoss()(torch.tensor(preds_test),torch.tensor(ys_test)).item()
#         print('loss in runner :', loss_run, loss_run2)
#         print('nemidunam_runner : {}, {}'.format(model_trainer.evaluate_test['rmse_components'], model_trainer.evaluate_test['rmse']))

        # visualize_metrics(model_trainer, show_plot=False, save_plot=True)
        print('Train finished!')

    elif running_mode == 'finetune':
        model_fine_tuner = FineTuner('./config/fine_tune.yml')
        model_fine_tuner.finetune()
        #visualize_ts(model_fine_tuner, show_plot=False, save_plot=True)
        #visualize_metrics(model_fine_tuner, show_plot=False, save_plot=True)
        print('Fine Tune finished!')

    elif running_mode == 'test':
        model_tester = Tester('./config/test.yml')
        model_tester.test()
        visualize_ts(model_tester, show_plot=True, save_plot=True)
        print('Test finished!')

    elif running_mode == 'gridsearch':
        model_grid_searcher = GridSearcher(gridsearch_config_path='./config/grid_search.yml',
                                           gridsearch_space_config_path='./config/grid_search_space.yml')
        model_grid_searcher.gridsearch()
        print('Grid Search finished!')
