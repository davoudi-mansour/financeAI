import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, seq_len_in, num_features, num_outputs, seq_len_out=1):
        super(Linear, self).__init__()
        self.name = 'linear'
        self.linear = nn.Linear(seq_len_in * num_features, seq_len_out * num_outputs)
        self.num_outputs = num_outputs

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        x = src
        x = x.view(x.size(0), -1)
        pred = self.linear(x)
        pred = pred.view(pred.size(0), -1, self.num_outputs)

        return pred, trg_y
