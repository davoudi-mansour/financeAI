import os
import torch
import datetime

class EarlyStopping:

    def __init__(self, metric='loss', patience=10, delta=0, path_dir='./tmp', model_name='model'):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        self.metric = metric
        self.patience = patience
        self.delta = delta
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(path_dir, 'checkpoint_' + model_name + '_' + now + '.pt')
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, model, metric_tracker):
        val_loss = metric_tracker.val_loss[self.metric][-1]
        epoch = len(metric_tracker.val_loss[self.metric])
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(epoch, val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(epoch, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, epoch, val_loss, model):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_loss': val_loss
        }, self.path)