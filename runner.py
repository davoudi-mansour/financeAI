import sys
from trainer import Trainer
from fine_tuner import FineTuner
from tester import Tester
from grid_searcher import GridSearcher
from visualization import visualize_ts, visualize_metrics

if __name__ == '__main__':
    running_mode = sys.argv[1]
    if running_mode in ('train', 'test', 'finetune', 'gridsearch'):
        print(f'{running_mode} started!')
    else:
        exit(f'running mode "{running_mode}" is not supported!')

    if running_mode == 'train':
        model_trainer = Trainer('./config/train.yml')
        model_trainer.train()
        visualize_ts(model_trainer, show_plot=True, save_plot=True)
        for y_id, target_column in enumerate(model_trainer.params['target_columns']):
            for i_component in range(0, model_trainer.params['seq_len_out']):
                visualize_ts(model_trainer, show_plot=False, save_plot=True, i_component=i_component, y_id=y_id)
        visualize_metrics(model_trainer, show_plot=False, save_plot=True)
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