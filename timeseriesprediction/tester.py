from trainer import Trainer
import torch

class Tester(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ft_checkpoint = torch.load(self.params['save_model_path'])
        self.model.load_state_dict(ft_checkpoint)
        pass

    def test(self):
        self.evaluate_test = self.evaluate(self.test_loader)
        self.evaluate_train = self.evaluate_test
        return True