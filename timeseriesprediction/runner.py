import sys
from trainer import Trainer
from fine_tuner import FineTuner
from tester import Tester
from grid_searcher import GridSearcher
from param_tuner import ParamTuner
from visualization import visualize_ts, visualize_metrics
from metrics import mse, rmse, mae, smape


if __name__ == '__main__':
    running_mode = sys.argv[1]
    if running_mode in ('train', 'test', 'finetune', 'gridsearch', 'paramtune'):
        print(f'{running_mode} started!')
    else:
        exit(f'running mode "{running_mode}" is not supported!')

    if running_mode == 'train':
        model_trainer = Trainer('./config/train.yml')
        model_trainer.train()
        #-----------------------------------------------------------------------------------------------------#
        print(f'Test Results Normalized: RMSE={model_trainer.evaluate_test["rmse"]} | MAE={model_trainer.evaluate_test["mae"]} | MAPE={model_trainer.evaluate_test["mape"]} | SMAPE={model_trainer.evaluate_test["smape"]} | loss={model_trainer.evaluate_test["loss"]}')

        print(f'RMSE Components Normalized : {model_trainer.evaluate_test["rmse_components"]}')
        print(f'MAE Components Normalized : {model_trainer.evaluate_test["mae_components"]}')
        print(f'MAPE Components Normalized : {model_trainer.evaluate_test["mape_components"]}')
        print(f'SMAPE Components Normalized : {model_trainer.evaluate_test["smape_components"]}')
        print(f'loss Components Normalized : {model_trainer.evaluate_test["loss_components"]}')

        #-----------------------------------------------------------------------------------------------------#
        for y_id, target_column in enumerate(model_trainer.params['target_columns']):
            for i_component in range(0, model_trainer.params['seq_len_out']-1):
                visualize_ts(model_trainer, show_plot=False, save_plot=False, i_component=i_component, y_id=y_id)

        visualize_metrics(model_trainer, show_plot=False, save_plot=True)
        print('Train finished!')

    elif running_mode == 'finetune':
        model_fine_tuner = FineTuner('./config/fine_tune.yml')
        model_fine_tuner.finetune()
        print('Fine Tune finished!')

    elif running_mode == 'test':
        model_tester = Tester('./config/test.yml')
        model_tester.test()
        visualize_ts(model_tester, show_plot=False, save_plot=False)
        print('Test finished!')

    elif running_mode == 'gridsearch':
        model_grid_searcher = GridSearcher(gridsearch_config_path='./config/grid_search.yml',
                                           gridsearch_space_config_path='./config/grid_search_space.yml')
        model_grid_searcher.gridsearch()
        print('Grid Search finished!')

    elif running_mode == 'paramtune':
        model_param_tuner = ParamTuner(gridsearch_config_path='./config/grid_search.yml',
                                        gridsearch_space_config_path='./config/grid_search_space.yml')
        model_param_tuner.paramtune()
        print('Param Tune finished!')
