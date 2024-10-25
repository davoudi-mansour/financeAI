import torch.nn as nn
from .utils import series_decomp


class Dlinear(nn.Module):
    def __init__(self, seq_len, pred_len, num_features, num_outputs):
        super(Dlinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_outputs = num_outputs

        # Decomposition Kernel Size
        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)

        self.Linear_Seasonal = nn.Linear(seq_len*num_features, pred_len*num_outputs)
        self.Linear_Trend = nn.Linear(seq_len*num_features, pred_len*num_outputs)

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        x = src
        seasonal_init, trend_init = self.decomposition(x)

        seasonal_init = seasonal_init.view(seasonal_init.size(0), -1)
        trend_init = trend_init.view(trend_init.size(0), -1)
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        pred = x.view(x.size(0), -1, self.num_outputs)
        return pred, trg_y
