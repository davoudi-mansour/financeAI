from trainer import Trainer
import torch

class FineTuner(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ft_checkpoint = torch.load(self.params['save_model_path'])
        self.model.load_state_dict(ft_checkpoint)
        pass
    def finetune(self):
        result = self.train()
        return result

