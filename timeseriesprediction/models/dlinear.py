import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import series_decomp

class Dlinear(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, num_outputs, individual, channel):
        super(Dlinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_outputs = num_outputs

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = channel

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(seq_len*num_features, pred_len*num_outputs))
                self.Linear_Trend.append(nn.Linear(seq_len*num_features, pred_len*num_outputs))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(seq_len*num_features, pred_len*num_outputs)
            self.Linear_Trend = nn.Linear(seq_len*num_features, pred_len*num_outputs)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        x = src
        seasonal_init, trend_init = self.decompsition(x)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_init = seasonal_init.view(seasonal_init.size(0), -1)
            trend_init = trend_init.view(trend_init.size(0), -1)
            # print("input size: ", seasonal_init.size())
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            # print("output size: ", seasonal_output.size())
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        pred = x.view(x.size(0), -1, self.num_outputs)
        return pred, trg_y
