import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from darts import TimeSeries
from torch.utils.data.dataloader import default_collate
import datetime
import pandas as pd

pd.options.mode.chained_assignment = None


class TimeSeriesDataset:
    def __init__(self, path, datetime_column, target_columns, input_columns, time_freq, seq_len_in,
                 seq_len_out,
                 seq_len_dec, batch_size, DEVICE, train_portion):
        self.path = path
        self.datetime_column = datetime_column
        self.target_columns = target_columns
        for target_column in target_columns:
            if target_column in input_columns:
                input_columns.remove(target_column)
        self.input_columns = input_columns
        self.input_column2x_id = {x_name: "x_" + str(i) for i, x_name in enumerate(self.input_columns)}
        self.input_column2x_id.update({x_name: "xy_" + str(i) for i, x_name in enumerate(self.target_columns)})
        self.x_id2input_column = {"x_" + str(i): x_name for i, x_name in enumerate(self.input_columns)}
        self.x_id2input_column.update({"xy_" + str(i): x_name for i, x_name in enumerate(self.target_columns)})

        self.target_column2y_id = {y_name: "y_" + str(i) for i, y_name in enumerate(self.target_columns)}
        self.y_id2target_column = {"y_" + str(i): y_name for i, y_name in enumerate(self.target_columns)}

        self.time_freq = time_freq
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.seq_len_dec = seq_len_dec
        self.batch_size = batch_size
        self.DEVICE = DEVICE
        self.train_portion = train_portion
        if self.train_portion == 1:
            self.mode = 'train'
        elif self.train_portion == 0:
            self.mode = 'test'
        else:
            self.mode = 'train_test'

        self.dataset_df = self.load_dataset_df()
        pass

    def get_data_loaders(self):
        df = self.dataset_df.sort_values(self.datetime_column).copy()
        for k, v in self.input_column2x_id.items():
            df[v] = df.loc[:, k]

        for k, v in self.target_column2y_id.items():
            df[v] = df.loc[:, k]

        df['ds'] = df.loc[:, self.datetime_column]

        df = df.drop(
            columns=list(set([self.datetime_column] + list(self.target_column2y_id.keys()) + list(
                self.input_column2x_id.keys()))))
        train, val = None, None
        norm = None

        if self.mode == 'train':
            train, val = df, None
        elif self.mode == 'test':
            train, val = None, df
        elif self.mode == 'train_test':
            train_nums = int(len(df) * self.train_portion)
            train, val = df[:train_nums], df[train_nums - self.seq_len_in:]

        train_loader, test_loader = None, None
        train_normal, val_normal = None, None
        if 'train' in self.mode:
            train_dataset = TimeSeriesDatasetEncDec(
                data_target=train[list(self.y_id2target_column.keys())],
                data_input=train[list(self.x_id2input_column.keys())],
                seq_len_in=self.seq_len_in,
                seq_len_out=self.seq_len_out,
                seq_len_dec=self.seq_len_dec)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=lambda x: tuple(x_.to(self.DEVICE) for x_ in default_collate(x)))
        if 'test' in self.mode:
            test_dataset = TimeSeriesDatasetEncDec(
                data_target=val[list(self.y_id2target_column.keys())],
                data_input=val[list(self.x_id2input_column.keys())],
                seq_len_in=self.seq_len_in,
                seq_len_out=self.seq_len_out,
                seq_len_dec=self.seq_len_dec)

            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                     collate_fn=lambda x: tuple(x_.to(self.DEVICE) for x_ in default_collate(x)))

        return train_loader, test_loader, norm, train, train_normal, val, val_normal

    def replace_nan(self, ts):

        arr_1 = np.array(ts.values()).reshape(-1, ts.values().shape[0])[0]
        mask = np.isnan(arr_1)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        arr_1[mask] = arr_1[idx[mask]]
        arr_2 = arr_1.reshape(ts.values().shape[0], 1)
        return arr_2

    def load_dataset_df(self):
        df = pd.read_csv(self.path)
        load_columns = [self.datetime_column] + self.target_columns + self.input_columns
        load_columns = list(set(load_columns))
        df = df[load_columns]
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column], errors='coerce')
        df[self.datetime_column] = df[self.datetime_column].dt.tz_localize(None)
        df = df.sort_values(self.datetime_column)
        df = df.fillna(method='ffill')

        tmp_df = df.groupby(pd.Grouper(key=self.datetime_column, freq=self.time_freq)).mean()
        dataset_df = tmp_df.reset_index()

        return dataset_df


class TimeSeriesDatasetEncDec(Dataset):
    def __init__(self, data_input, data_target, seq_len_in, seq_len_out=1,
                 seq_len_dec=1):
        self.data_input = data_input
        self.data_target = data_target
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.seq_len_dec = seq_len_dec

    def __len__(self):
        return len(self.data_target) - self.seq_len_in - self.seq_len_out + 1

    def __getitem__(self, index):
        sequence_in = torch.Tensor(self.data_input[index: index + self.seq_len_in + self.seq_len_out].values)
        sequence_out = torch.Tensor(self.data_target[index: index + self.seq_len_in + self.seq_len_out].values)

        src, trg, trg_y, trg_teacher_forcing = self.get_src_trg(sequence_in, sequence_out)

        return (src, trg, trg_y, trg_teacher_forcing)

    def get_src_trg(
            self,
            sequence_in: torch.Tensor,
            sequence_out: torch.Tensor,
    ):
        sequence_in_len = sequence_in.shape[0]
        sequence_out_len = sequence_out.shape[0]
        assert sequence_in_len == sequence_out_len, "Sequence_in length does not equal Sequence_out"
        assert sequence_in_len == self.seq_len_in + self.seq_len_out, "in and out Sequence length does not equal (seq_len_in + seq_len_out)"

        src = sequence_in[:self.seq_len_in]
        trg = sequence_out[self.seq_len_in - self.seq_len_dec:self.seq_len_in]
        trg_y = sequence_out[self.seq_len_in:]
        trg_teacher_forcing = sequence_out[self.seq_len_in:self.seq_len_in + len(trg_y)]
        return src, trg, trg_y, trg_teacher_forcing