import torch.nn as nn


class LSTM_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len_in, seq_len_out, n_decoder_layers):
        super(LSTM_MLP, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = input_size
        self.output_size = output_size

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out

        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True, bidirectional=True)

        self.decoder_mlp = Decoder(
            [
                MLPLayer(self.seq_len_in, self.seq_len_out, hidden_size, num_layers, output_size) for l in range(n_decoder_layers)
            ]
        )


    def is_teacher_forcing(self):

        if self.epoch_portion < self.teacher_forcing:
            return True
        return False

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        encoder_input_seq, decoder_input_seq, trg_teacher_forcing = src, trg, trg_teacher_forcing

        encoder_outputs, encoder_hidden = self.encoder_lstm(encoder_input_seq)
        outputs = encoder_outputs[:, encoder_outputs.shape[1] - 1:, :]
        pred = self.decoder_mlp(outputs)
        pred = pred.view(pred.size(0), -1, self.output_size)
        return pred, trg_y

class Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLPLayer(nn.Module):
    def __init__(self, seq_len, pred_len, hidden_size, num_layers, num_outputs):
        super().__init__()
        self.l1 = nn.Linear(hidden_size*num_layers, hidden_size)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(hidden_size, pred_len*num_outputs)

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x