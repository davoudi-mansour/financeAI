# encoding=utf-8
import torch
import torch.nn as nn
from .utils import series_decomp
import torch.nn.functional as F
import numpy as np


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class DMEAformer(nn.Module):
    def __init__(self, seq_len, pred_len, S, n_encoder_layers, n_decoder_layers, hidden_size, num_features, num_outputs):
        super(DMEAformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_outputs = num_outputs

        # self.channels = enc_in
        self.decomp1 = series_decomp(25)
        self.encoder1 = Encoder(
            [LEncoderLayer(self.seq_len, S) for l in range(n_encoder_layers)]
        )
        self.encoder2 = Encoder(
            [LEncoderLayer(self.seq_len, S) for l in range(n_encoder_layers)]
        )

        self.decoder1 = Decoder(
            [
                MLPLayer(self.seq_len if l == 0 else self.pred_len, self.pred_len, hidden_size, num_features, num_outputs) for l in range(n_decoder_layers)
            ]
        )

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        # x: [Batch, Input length, Channel]
        x = src
        # rev = RevIN(x.size(2)).cuda()
        # x = rev(x, 'norm')
        seasonal_init, trend_init = self.decomp1(x)
        seasonal1 = self.encoder1(seasonal_init)
        trend1 = self.encoder2(trend_init)
        out = seasonal1.permute(0, 2, 1) + trend1.permute(0, 2, 1)
        out = out.view(out.size(0), -1)
        x = self.decoder1(out)
        # x = rev(x, 'denorm')
        pred = x.view(x.size(0), -1, self.num_outputs)
        return pred, trg_y


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LEncoderLayer(nn.Module):
    def __init__(self, seq_len, S, norm='ln'):
        super().__init__()
        self.seq_len = seq_len

        # self.attn = nn.LSTM(seq_len, hidden_size)
        self.attn = TemporalExternalAttn(seq_len, S)
        self.drop1 = nn.Dropout(0.2)
        self.feed2 = nn.Linear(seq_len, seq_len)
        self.norm = norm
        if norm == 'ln':
            self.norm1 = nn.LayerNorm(seq_len)
            self.norm2 = nn.LayerNorm(seq_len)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

    def forward(self, x):
        # x = self.feed1(x)
        attn = self.attn(x.permute(0, 2, 1))
        x = x + attn.permute(0, 2, 1)
        if self.norm == 'ln':
            x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm1(x.permute(0, 2, 1))
        x = x + self.feed2(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.norm == 'ln':
            x = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm2(x.permute(0, 2, 1))
        return x


class TemporalExternalAttn(nn.Module):
    def __init__(self, d_model, S=256):
        super().__init__()

        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        # attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model
        return out


class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, seq_len, pred_len, hidden_size, num_features, num_outputs):
        super().__init__()
        self.l1 = nn.Linear(seq_len*num_features, hidden_size)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(hidden_size, pred_len*num_outputs)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x
