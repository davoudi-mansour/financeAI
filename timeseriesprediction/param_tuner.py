import yaml
import numpy as np
import itertools
from trainer import Trainer


class ParamTuner:
    def __init__(self, gridsearch_config_path, gridsearch_space_config_path):
        with open(gridsearch_config_path, "r") as stream:
            self.gridsearch_config = yaml.safe_load(stream)

        with open(gridsearch_space_config_path, "r") as stream:
            self.gridsearch_space = yaml.safe_load(stream)

    def paramtune(self):
        # initialize variables for storing the best hyperparameters and performance
        best_params = {}
        best_score = np.inf

        hyperparams = self.gridsearch_space
        model_type = hyperparams['model']
        model_hyperparams = self.gridsearch_space['model_hyper_params'][model_type[0]]
        hyperparams.pop('model_hyper_params')
        hyperparams.update(model_hyperparams)

        params_list = list(hyperparams.values())
        keys = list(hyperparams.keys())

        params = [[item[0]] for item in params_list]
        for i in range(1, len(params_list)):
            # print(params)
            if len(params_list[i]) > 1:
                params[i] = params_list[i]
                # print('params : ', params)
                states = list(itertools.product(*params))
        #         # print('states : ', states)
                best_state_score = np.inf
                best_state = 0
                for j in range(0, len(states)):
                    model_params = {k: v for k, v in zip(keys, states[j])}
                    print('model_params : ', model_params)

                    score = self.train_with_params(model_params)

                    if score < best_score:
                        best_score = score
                        best_params = model_params

                    if score < best_state_score:
                        best_state_score = score
                        best_state = j

                params[i] = [params[i][best_state]]
            print('#####################################################################')

        with open(self.gridsearch_config['gridsearch_output_path'], 'w') as outfile:
            yaml.dump(best_params, outfile, default_flow_style=False)

        print('best_score =', best_score)
        print('best_params =', best_params)

        return best_params, best_score
    def train_with_params(self, params):
        trainer_params = params.copy()
        trainer_params.update(self.gridsearch_config)
        with open('./tmp/tmp_tuner.yml', 'w') as outfile:
            yaml.dump(trainer_params, outfile, default_flow_style=False)

        model_trainer = Trainer(model_config_path='./tmp/tmp_tuner.yml')
        model_trainer.train()
        score = model_trainer.evaluate_test['smape']
        print('score : ', score)
        return score
