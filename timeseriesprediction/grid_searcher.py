import yaml
import numpy as np
import itertools
from trainer import Trainer


class GridSearcher:
    def __init__(self, gridsearch_config_path, gridsearch_space_config_path):
        with open(gridsearch_config_path, "r") as stream:
            self.gridsearch_config = yaml.safe_load(stream)

        with open(gridsearch_space_config_path, "r") as stream:
            self.gridsearch_space = yaml.safe_load(stream)

    def gridsearch(self):
        # initialize variables for storing the best hyperparameters and performance
        best_params = {}
        best_score = np.inf

        # get all possible combinations of hyperparameters
        hyperparams = []
        for k, v in self.gridsearch_space.items():
            if k != 'model_hyper_params':
                hyperparams.append(v)
        hyperparams = list(itertools.product(*hyperparams))
        results = []
        # iterate over all combinations of hyperparameters
        for params in hyperparams:
            # construct model
            model_type = params[-1]
            model_hyperparams = self.gridsearch_space['model_hyper_params'][model_type]
            model_params = {k: v for k, v in zip(self.gridsearch_space.keys(), params)}
            model_params.update(model_hyperparams)

            # generate all possible combinations of hyperparameters in model_hyperparams
            model_hyperparams = self.gridsearch_space['model_hyper_params'][model_type]
            hyperparam_values = list(itertools.product(*model_hyperparams.values()))
            for hyperparams in hyperparam_values:
                model_params.update(dict(zip(model_hyperparams.keys(), hyperparams)))
                print("##############")
                print("##############")
                print("##############")
                print("model_params: ", model_params)

                score = self.train_with_params(model_params)
                # score = 1
                if score < best_score:
                    best_score = score
                    best_params = model_params

        with open(self.gridsearch_config['gridsearch_output_path'], 'w') as outfile:
            yaml.dump(best_params, outfile, default_flow_style=False)

        print('best_score =', best_score)
        print('best_params =', best_params)

        return best_params, best_score

    def train_with_params(self, params):
        trainer_params = params.copy()
        trainer_params.update(self.gridsearch_config)
        with open('tmp/tmp_grid_search.yml', 'w') as outfile:
            yaml.dump(trainer_params, outfile, default_flow_style=False)

        model_trainer = Trainer(model_config_path='tmp/tmp_grid_search.yml')
        model_trainer.train()
        score = model_trainer.evaluate_test['smape']
        print("score: ", score)
        return score