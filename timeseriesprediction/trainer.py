import os
import yaml
import tqdm
import torch
import shutil
import numpy as np
from torch import nn
from Norm import Norm, RevIN
import torch.optim as optim
from MetricTracker import MetricTracker
from EarlyStopping import EarlyStopping
from models.get_model import get_model
from metrics import smape, mae, rmse, mse, mape
from dataset import TimeSeriesDataset


class Trainer:
    def __init__(self, model_config_path, params=None, use_scheduler=False):
        if model_config_path is not None:
            with open(model_config_path, "r") as stream:
                self.params = yaml.safe_load(stream)
        else:
            self.params = params
        self.params['num_features'] = len(self.params['input_columns'])
        self.params['num_outputs'] = len(self.params['target_columns'])
        self.params["DEVICE"] = torch.device(self.params["DEVICE"])
        self.ts_ds = TimeSeriesDataset(path=self.params['dataset_path'],
                                       datetime_column=self.params['datetime_column'],
                                       input_columns=self.params['input_columns'],
                                       target_columns=self.params['target_columns'],
                                       time_freq=self.params['time_freq'],
                                       seq_len_in=self.params['seq_len_in'],
                                       seq_len_out=self.params['seq_len_out'],
                                       seq_len_dec=self.params['seq_len_dec'],
                                       batch_size=self.params['batch_size'],
                                       DEVICE=self.params['DEVICE'],
                                       train_portion=self.params['train_portion'],
                                       )
        self.train_loader, self.test_loader, self.normalizer, self.data_train, self.data_train_normal, self.data_val, self.data_val_normal = self.ts_ds.get_data_loaders()
        self.model = get_model(self.params)

        self.metric_tracker = MetricTracker(metrics=['rmse', 'mae', 'smape', 'loss'])

        self.early_stopping = EarlyStopping(metric='loss',
                                            patience=self.params['early_stopping_patience'],
                                            path_dir=self.params['tmp_directory'],
                                            delta=0,
                                            )
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(self.params['lr']))
        self.scheduler = None
        if use_scheduler:
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.999)

        self.loss_fn_mean = nn.MSELoss()

        self.evaluate_train = None
        self.evaluate_test = None

    def train_one_epoch(self, epoch, pbar=None):
        preds, ys = [], []
        losses = []
        len_train_loader = len(self.train_loader)
        train_mean_loss = 0
        for batch_idx, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()
            src, trg, trg_y, trg_teacher_forcing = batch
            norm_src = Norm(src, norm_type=self.params['norm_type'])
            norm_trg = Norm(trg, norm_type=self.params['norm_type'])
            norm_trg_y = Norm(trg_y, norm_type=self.params['norm_type'])
            norm_trg_teacher_forcing = Norm(trg_teacher_forcing, norm_type=self.params['norm_type'])

            pred, y = self.model(norm_src.normalize(src),
                                 norm_trg.normalize(trg),
                                 norm_trg_y.normalize(trg_y),
                                 norm_trg_teacher_forcing.normalize(trg_teacher_forcing),
                                 epoch_portion=epoch / self.params['num_epochs'])

            preds.append(pred.cpu().detach().numpy())
            ys.append(y.cpu().detach().numpy())
            loss = self.loss_fn_mean(pred, y)
            backward_loss = loss
            backward_loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            train_mean_loss = sum(losses) / len(losses)
            if pbar:
                pbar.set_description(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len_train_loader}| Loss = {train_mean_loss:.3E}")
                pbar.update(1)

        if self.scheduler is not None:
            self.scheduler.step()

        return train_mean_loss

    def train(self):

        for epoch in range(self.params['num_epochs']):
            len_train_loader = len(self.train_loader)
            pbar = None
            if self.params['displaying_progress_bar']:
                pbar = tqdm.tqdm(total=len_train_loader, desc="Processing items...")

            train_mean_loss = self.train_one_epoch(epoch, pbar)

            self.logging_end_of_epoch(epoch, train_mean_loss, pbar)

            if self.early_stopping.early_stop:
                print("\t *****")
                print("\t  ***")
                print(f"Early stopping at {epoch}/{self.params['num_epochs']}")
                print("\t  ***")
                print("\t *****")
                break

        if self.early_stopping is not None:
            checkpoint = torch.load(self.early_stopping.path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.evaluate_train = self.evaluate(self.train_loader)
        if 'test' in self.ts_ds.mode:
            self.evaluate_test = self.evaluate(self.test_loader)
        else:
            self.evaluate_test = self.evaluate_train

        torch.save(self.model.state_dict(), self.params['save_model_path'])
        self.remove_tmp_files()
        return True

    def evaluate(self, data_loader):

        self.model.eval()
        with torch.no_grad():
            preds, ys = [], []
            preds_denorm, ys_denorm = [], []
            for batch in data_loader:
                src, trg, trg_y, trg_teacher_forcing = batch
                norm_src = Norm(src, norm_type=self.params['norm_type'])
                norm_trg = Norm(trg, norm_type=self.params['norm_type'])
                norm_trg_y = Norm(trg_y, norm_type=self.params['norm_type'])
                norm_trg_teacher_forcing = Norm(trg_teacher_forcing, norm_type=self.params['norm_type'])

                pred, y = self.model(norm_src.normalize(src),
                                     norm_trg.normalize(trg),
                                     norm_trg_y.normalize(trg_y),
                                     norm_trg_teacher_forcing.normalize(trg_teacher_forcing),
                                     epoch_portion=1)

                pred_denorm = norm_trg_y.denormalize(pred)
                y_denorm = norm_trg_y.denormalize(y)
                preds.append(pred.cpu().detach().numpy())
                ys.append(y.cpu().detach().numpy())
                preds_denorm.append(pred_denorm.cpu().detach().numpy())
                ys_denorm.append(y_denorm.cpu().detach().numpy())

            if len(preds) > 0:
                preds = np.concatenate(preds)
            else:
                preds = np.array(preds)
            if len(ys) > 0:
                ys = np.concatenate(ys)
            else:
                ys = np.array(ys)

            if len(preds_denorm) > 0:
                preds_denorm = np.concatenate(preds_denorm)
            else:
                preds_denorm = np.array(preds_denorm)
            if len(ys_denorm) > 0:
                ys_denorm = np.concatenate(ys_denorm)
            else:
                ys_denorm = np.array(ys_denorm)

        loss = nn.MSELoss()(torch.Tensor(preds), torch.Tensor(ys)).item()
        return_dict = dict()
        return_dict['preds'] = preds_denorm
        return_dict['ys'] = ys_denorm
        return_dict['rmse'] = rmse(preds, ys, axis=None)
        return_dict['mae'] = mae(preds, ys, axis=None)
        return_dict['mape'] = mape(preds, ys, axis=None)
        return_dict['smape'] = smape(preds, ys, axis=None)
        return_dict['rmse_components'] = rmse(preds, ys, axis=0)
        return_dict['mae_components'] = mae(preds, ys, axis=0)
        return_dict['mape_components'] = mape(preds, ys, axis=0)
        return_dict['smape_components'] = smape(preds, ys, axis=0)
        return_dict['loss_components'] = mse(preds, ys, axis=0)
        return_dict['loss'] = loss
        return return_dict

    def logging_end_of_epoch(self, epoch, train_mean_loss, pbar):
        len_train_loader = len(self.train_loader)

        evaluate_train = self.evaluate(self.train_loader)
        if 'test' in self.ts_ds.mode:
            evaluate_test = self.evaluate(self.test_loader)
        else:
            evaluate_test = evaluate_train

        if pbar:
            pbar.set_description(
                f"Epoch {epoch + 1}, Batch {len_train_loader}/{len_train_loader} | Train Loss = {train_mean_loss:.3E}, Test Loss = {evaluate_test['loss']:.3E}")

        if self.params['print_interval'] and (
                epoch % self.params['print_interval'] == 0 or (epoch + 1) == self.params['num_epochs']):
            print("----------")
            print(
                f'Epoch {epoch + 1}/{self.params["num_epochs"]} | Train Loss: {train_mean_loss:.2E} | Test Loss: {evaluate_test["loss"]:.2E}')

            print(
                f'Train results: RMSE={evaluate_train["rmse"]:.2E} | MAE={evaluate_train["mae"]:.2E} | SMAPE={evaluate_train["smape"]:.2f} | loss={evaluate_train["loss"]:.2E}')
            print(
                f'TEST results: RMSE={evaluate_test["rmse"]:.2E} | MAE={evaluate_test["mae"]:.2E} | SMAPE={evaluate_test["smape"]:.2f} | loss={evaluate_test["loss"]:.2E}')

        if self.metric_tracker is not None:
            self.metric_tracker.log(evaluate_train, evaluate_test, evaluate_test)

            if self.early_stopping is not None:
                self.early_stopping.step(self.model, self.metric_tracker)
        return 0

    def remove_tmp_files(self, ):
        folder = self.params['tmp_directory']
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
