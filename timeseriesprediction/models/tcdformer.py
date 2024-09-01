# from layers.TCDformer_EncDec import EncoderLayer, Encoder, DecoderLayer, Decoder, AttentionLayer, series_decomp, series_decomp_multi
import torch.nn as nn
import torch
# from layers.Embed import DataEmbedding
# from layers.Attention import WaveletAttention, FourierAttention, FullAttention
# from layers.RevIN import RevIN
# from layers.LLSA import LLSA
import torch.nn.functional as F
import math
import numpy as np
from math import sqrt
from .utils import get_filter, series_decomp, series_decomp_multi
from torch import Tensor
from typing import List, Tuple


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
        #                                             freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
        #     d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
"""
Full Attention
"""
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=3, scale=None, attention_dropout=0.1, T=1, activation='softmax',
                 output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.activation = activation
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.T = T

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale

        if self.activation == 'softmax':
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)

                scores.masked_fill_(attn_mask.mask, -np.inf)

            A = self.dropout(torch.softmax(scores / self.T, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)

        elif self.activation == 'linear':
            V = torch.einsum("bhls,bshd->blhd", scores, values)

        elif self.activation == 'linear_norm':
            mins = scores.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, scores.shape[3])
            scores = scores - mins + 1e-8

            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, 0)

            sums = scores.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, scores.shape[3])
            scores /= sums
            V = torch.einsum("bhls,bshd->blhd", scores, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

"""
Fourier Attention
"""
class FourierAttention(nn.Module):
    def __init__(self, T=1, activation='softmax', output_attention=False):
        super(FourierAttention, self).__init__()
        print(' fourier enhanced cross attention used!')
        """
        1D Fourier Cross Attention layer. It does FFT, linear transform, attention mechanism and Inverse FFT.    
        """
        self.activation = activation
        self.output_attention = output_attention
        self.T = T

    def forward(self, q, k, v, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        _, S, H, E = k.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        xq_ft_ = torch.fft.rfft(xq, dim=-1, norm='ortho')
        xk_ft_ = torch.fft.rfft(xk, dim=-1, norm='ortho')
        xv_ft_ = torch.fft.rfft(xv, dim=-1, norm='ortho')

        xqk_ft = torch.einsum("bhex,bhey->bhxy", xq_ft_, torch.conj(xk_ft_)) / sqrt(E)

        if self.activation == 'softmax':
            xqk_ft = torch.softmax(xqk_ft.abs() / self.T, dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear':
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            mins_imag = xqk_ft.imag.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag = xqk_ft.imag - mins_imag
            sums_imag = xqk_ft_imag.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_imag /= sums_imag

            xqkv_ft_real = torch.einsum("bhxy,bhey->bhex", xqk_ft_real, xv_ft_.real)
            xqkv_ft_imag = torch.einsum("bhxy,bhey->bhex", xqk_ft_imag, xv_ft_.imag)
            xqkv_ft = torch.complex(xqkv_ft_real, xqkv_ft_imag)

        elif self.activation == 'linear_norm_abs':
            xqk_ft = xqk_ft.abs() / xqk_ft.abs().sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        elif self.activation == 'linear_norm_real':
            mins_real = xqk_ft.real.min(dim=-1)[0].unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real = xqk_ft.real - mins_real
            sums_real = xqk_ft_real.sum(dim=-1).unsqueeze(-1).expand(-1, -1, -1, xqk_ft.shape[3])
            xqk_ft_real /= sums_real

            xqk_ft = torch.complex(xqk_ft_real, torch.zeros_like(xqk_ft_real))
            xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xv_ft_)

        out = torch.fft.irfft(xqkv_ft, n=L, dim=-1, norm='ortho').permute(0, 3, 1, 2)

        if self.output_attention == False:
            return (out, None)
        else:
            return (out, (xqk_ft_real, xqk_ft_imag))


"""
Wavelet Attention
"""
class WaveletAttention(nn.Module):
    """
    1D Multiwavelet Cross Attention layer.
    """

    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes, c=64,
                 k=8, ich=512,
                 L=0,
                 base='legendre',
                 mode_select_method='random',
                 initializer=None, activation='tanh',
                 **kwargs):
        super(WaveletAttention, self).__init__()
        print('base', base)
        print('Wavelet attention used!')
        self.c = c
        self.k = k
        self.L = L
        H0, H1, G0, G1, PHI0, PHI1 = get_filter(base, k)
        H0r = H0 @ PHI0
        G0r = G0 @ PHI0
        H1r = H1 @ PHI1
        G1r = G1 @ PHI1

        H0r[np.abs(H0r) < 1e-8] = 0
        H1r[np.abs(H1r) < 1e-8] = 0
        G0r[np.abs(G0r) < 1e-8] = 0
        G1r[np.abs(G1r) < 1e-8] = 0
        self.max_item = 3

        self.attn1 = WaveletCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn2 = WaveletCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn3 = WaveletCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.attn4 = WaveletCrossAttention(in_channels=in_channels, out_channels=out_channels, seq_len_q=seq_len_q,
                                            seq_len_kv=seq_len_kv, modes=modes, activation=activation,
                                            mode_select_method=mode_select_method)
        self.T0 = nn.Linear(k, k)
        self.register_buffer('ec_s', torch.Tensor(
            np.concatenate((H0.T, H1.T), axis=0)))
        self.register_buffer('ec_d', torch.Tensor(
            np.concatenate((G0.T, G1.T), axis=0)))

        self.register_buffer('rc_e', torch.Tensor(
            np.concatenate((H0r, G0r), axis=0)))
        self.register_buffer('rc_o', torch.Tensor(
            np.concatenate((H1r, G1r), axis=0)))

        self.Lk = nn.Linear(ich, c * k)
        self.Lq = nn.Linear(ich, c * k)
        self.Lv = nn.Linear(ich, c * k)
        self.out = nn.Linear(c * k, ich)
        self.modes1 = modes

    def forward(self, q, k, v, mask=None):
        B, N, H, E = q.shape  # (B, N, H, E) torch.Size([3, 768, 8, 2])
        _, S, _, _ = k.shape  # (B, S, H, E) torch.Size([3, 96, 8, 2])

        q = q.view(q.shape[0], q.shape[1], -1)
        k = k.view(k.shape[0], k.shape[1], -1)
        v = v.view(v.shape[0], v.shape[1], -1)
        q = self.Lq(q)
        q = q.view(q.shape[0], q.shape[1], self.c, self.k)
        k = self.Lk(k)
        k = k.view(k.shape[0], k.shape[1], self.c, self.k)
        v = self.Lv(v)
        v = v.view(v.shape[0], v.shape[1], self.c, self.k)

        if N > S:
            zeros = torch.zeros_like(q[:, :(N - S), :]).float()
            v = torch.cat([v, zeros], dim=1)
            k = torch.cat([k, zeros], dim=1)
        else:
            v = v[:, :N, :, :]
            k = k[:, :N, :, :]

        ns = math.floor(np.log2(N))
        nl = pow(2, math.ceil(np.log2(N)))
        extra_q = q[:, 0:nl - N, :, :]
        extra_k = k[:, 0:nl - N, :, :]
        extra_v = v[:, 0:nl - N, :, :]
        q = torch.cat([q, extra_q], 1)
        k = torch.cat([k, extra_k], 1)
        v = torch.cat([v, extra_v], 1)

        Ud_q = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_k = torch.jit.annotate(List[Tuple[Tensor]], [])
        Ud_v = torch.jit.annotate(List[Tuple[Tensor]], [])

        Us_q = torch.jit.annotate(List[Tensor], [])
        Us_k = torch.jit.annotate(List[Tensor], [])
        Us_v = torch.jit.annotate(List[Tensor], [])

        Ud = torch.jit.annotate(List[Tensor], [])
        Us = torch.jit.annotate(List[Tensor], [])

        # decompose
        for i in range(ns - self.L):
            # print('q shape',q.shape)
            d, q = self.wavelet_transform(q)
            Ud_q += [tuple([d, q])]
            Us_q += [d]
        for i in range(ns - self.L):
            d, k = self.wavelet_transform(k)
            Ud_k += [tuple([d, k])]
            Us_k += [d]
        for i in range(ns - self.L):
            d, v = self.wavelet_transform(v)
            Ud_v += [tuple([d, v])]
            Us_v += [d]
        for i in range(ns - self.L):
            dk, sk = Ud_k[i], Us_k[i]
            dq, sq = Ud_q[i], Us_q[i]
            dv, sv = Ud_v[i], Us_v[i]
            Ud += [self.attn1(dq[0], dk[0], dv[0], mask)[0] + self.attn2(dq[1], dk[1], dv[1], mask)[0]]
            Us += [self.attn3(sq, sk, sv, mask)[0]]
        v = self.attn4(q, k, v, mask)[0]

        # reconstruct
        for i in range(ns - 1 - self.L, -1, -1):
            v = v + Us[i]
            v = torch.cat((v, Ud[i]), -1)
            v = self.evenOdd(v)
        v = self.out(v[:, :N, :, :].contiguous().view(B, N, -1))
        return (v.contiguous(), None)

    def wavelet_transform(self, x):
        xa = torch.cat([x[:, ::2, :, :],
                        x[:, 1::2, :, :],
                        ], -1)
        d = torch.matmul(xa, self.ec_d)
        s = torch.matmul(xa, self.ec_s)
        return d, s

    def evenOdd(self, x):
        B, N, c, ich = x.shape  # (B, N, c, k)
        assert ich == 2 * self.k
        x_e = torch.matmul(x, self.rc_e)
        x_o = torch.matmul(x, self.rc_o)

        x = torch.zeros(B, N * 2, c, self.k,
                        device=x.device)
        x[..., ::2, :, :] = x_e
        x[..., 1::2, :, :] = x_o
        return x

class WaveletCrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, modes=16, activation='tanh',
                 mode_select_method='random'):
        super(WaveletCrossAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.activation = activation

    def forward(self, q, k, v, mask):
        B, L, E, H = q.shape

        xq = q.permute(0, 3, 2, 1)  # size = [B, H, E, L] torch.Size([3, 8, 64, 512])
        xk = k.permute(0, 3, 2, 1)
        xv = v.permute(0, 3, 2, 1)
        self.index_q = list(range(0, min(int(L // 2), self.modes1)))
        self.index_k_v = list(range(0, min(int(xv.shape[3] // 2), self.modes1)))

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]

        xk_ft_ = torch.zeros(B, H, E, len(self.index_k_v), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_k_v):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]
        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)

        xqkvw = xqkv_ft
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 3, 2, 1)
        # size = [B, L, H, E]
        return (out, None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads=2, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=512, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=512,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x_o, attn1 = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(x_o)
        x = self.norm1(x)

        x_o, attn2 = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(x_o)

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y), (attn1, attn2)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x, attns

class LLSA(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, window_size=5, hidden_size=10, change_point_threshold=0.1):
        super(LLSA, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.change_point_threshold = nn.Parameter(torch.tensor(change_point_threshold))
        self.linear = nn.Linear(window_size, hidden_size)
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
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
class TCDformer(nn.Module):
    def __init__(self,
                 version,
                 seq_len,
                 label_len,
                 pred_len,
                 e_layers,
                 d_layers,
                 enc_in,
                 dec_in,
                 c_out,
                 moving_avg,
                 d_model,
                 dropout,
                 modes,
                 activation,
                 window_size,
                 hidden_size,
                 output_attention=False,
                 output_stl=False
                 ):
        super(TCDformer, self).__init__()
        self.version = version
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.output_stl = output_stl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_outputs = c_out

        kernel_size = moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        self.enc_seasonal_embedding = DataEmbedding(enc_in, d_model, dropout)
        self.dec_seasonal_embedding = DataEmbedding(dec_in, d_model, dropout)

        if version == 'Wavelet':
            enc_self_attention = WaveletAttention(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=seq_len,
                                                  seq_len_kv=seq_len,
                                                  ich=d_model,
                                                  modes=modes,
                                                  activation=activation,
                                                  output_attention=output_attention)
            dec_self_attention = WaveletAttention(in_channels=d_model,
                                                  out_channels=d_model,
                                                  seq_len_q=seq_len // 2 + pred_len,
                                                  seq_len_kv=seq_len // 2 + pred_len,
                                                  ich=d_model,
                                                  modes=modes,
                                                  activation=activation,
                                                  output_attention=output_attention)
            dec_cross_attention = WaveletAttention(in_channels=d_model,
                                                   out_channels=d_model,
                                                   seq_len_q=seq_len // 2 + pred_len,
                                                   seq_len_kv=seq_len,
                                                   ich=d_model,
                                                   modes=modes,
                                                   activation=activation,
                                                   output_attention=output_attention)
        elif version == 'Fourier':
            enc_self_attention = FourierAttention(activation=activation,
                                                  output_attention=output_attention)
            dec_self_attention = FourierAttention(activation=activation,
                                                  output_attention=output_attention)
            dec_cross_attention = FourierAttention(activation=activation,
                                                   output_attention=output_attention)
        elif version == 'Time':
            enc_self_attention = FullAttention(False, activation=activation,
                                               attention_dropout=dropout,
                                               output_attention=output_attention)
            dec_self_attention = FullAttention(True, activation=activation,
                                               attention_dropout=dropout,
                                               output_attention=output_attention)
            dec_cross_attention = FullAttention(False, activation=activation,
                                                attention_dropout=dropout,
                                                output_attention=output_attention)

        self.seasonal_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        enc_self_attention,
                        d_model),
                    d_model,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.seasonal_decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        dec_self_attention,
                        d_model),
                    AttentionLayer(
                        dec_cross_attention,
                        d_model),
                    d_model,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True)
        )

        self.trend = nn.Sequential(
            nn.Linear(seq_len*enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len*c_out)
        )

        self.llsa_trend = LLSA(enc_in, window_size=window_size, hidden_size=hidden_size).to(
            self.device)


    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # zeros = torch.zeros([trg.shape[0], self.pred_len, trg.shape[2]]).to(self.device)  # cuda()
        seasonal_enc, trend_enc = self.decomp(src)
        seasonal_dec = F.pad(seasonal_enc[:, -self.label_len:, -2:], (0, 0, 0, self.pred_len))
        # seasonal_dec, trend_dec = self.decomp(trg)
        enc_out = self.enc_seasonal_embedding(seasonal_enc)
        enc_out, attn_e = self.seasonal_encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_seasonal_embedding(seasonal_dec)
        # print("enc_out: ", enc_out.size())
        # print("dec_out: ", dec_out.size())


        seasonal_out, attn_d = self.seasonal_decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        seasonal_out = seasonal_out[:, -self.pred_len:, :]

        # seasonal_ratio = seasonal_enc.abs().mean(dim=1) / seasonal_out.abs().mean(dim=1)
        # seasonal_ratio = seasonal_ratio.unsqueeze(1).expand(-1, self.pred_len, -1)

        # trend_enc = self.llsa_trend(trend_enc, 'norm')
        trend_out = self.trend(trend_enc.view(trend_enc.size(0), -1))
        # trend_out = self.llsa_trend(trend_out, 'denorm')
        trend_out = trend_out.view(trend_out.size(0), -1, self.num_outputs)
        # dec_out = trend_out + seasonal_ratio * seasonal_out

        dec_out = trend_out + seasonal_out

        # if self.output_attention:
        #     return dec_out, (attn_e, attn_d)
        # elif self.output_stl:
        #     return dec_out, trend_enc, seasonal_enc, trend_out, seasonal_ratio * seasonal_out
        # else:
        return dec_out, trg_y

