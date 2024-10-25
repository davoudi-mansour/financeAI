import torch.nn as nn

class MEAformer(nn.Module):

    def __init__(self, seq_len, pred_len, S, n_encoder_layers, n_decoder_layers, hidden_size,num_features, num_outputs):
        super(MEAformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_outputs = num_outputs

        self.encoder = Encoder(
            [LEncoderLayer(self.seq_len, S) for l in range(n_encoder_layers)]
        )

        self.decoder = Decoder(
            [
                MLPLayer(self.seq_len if l == 0 else self.pred_len, self.pred_len, hidden_size, num_features, num_outputs) for l in range(n_decoder_layers)
            ]
        )

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        x = src
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
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


