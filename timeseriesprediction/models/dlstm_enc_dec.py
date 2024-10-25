import torch
import torch.nn as nn
from .utils import series_decomp


class DEncoderDecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len_in, seq_len_out, seq_len_dec,
                 teacher_forcing):
        super(DEncoderDecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = input_size
        self.output_size = output_size

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.seq_len_dec = seq_len_dec
        self.teacher_forcing = teacher_forcing

        self.decomposition = series_decomp(25)

        self.encoder_seasonal_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                             batch_first=True, bidirectional=True)

        self.encoder_trend_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                          batch_first=True, bidirectional=True)

        self.decoder_lstm = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=2 * num_layers,
                                    batch_first=True)
        self.decoder_output_linear = nn.Linear(hidden_size, output_size)
        self.counter = 0
        self.seq_len_out_controlled = False
        self.epoch_portion = 0

    def is_teacher_forcing(self):

        if self.epoch_portion < self.teacher_forcing:
            return True
        return False

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        self.epoch_portion = epoch_portion
        encoder_input_seasonal, encoder_input_trend = self.decomposition(src)
        decoder_input_seq, trg_teacher_forcing = trg, trg_teacher_forcing
        self.counter += 1
        encoder_seasonal_outputs, encoder_seasonal_hidden = self.encoder_seasonal_lstm(encoder_input_seasonal)
        encoder_trend_outputs, encoder_trend_hidden = self.encoder_trend_lstm(encoder_input_trend)

        decoder_input = decoder_input_seq
        decoder_hidden = encoder_seasonal_hidden[0] + encoder_trend_hidden[0]
        decoder_cell = encoder_seasonal_hidden[1] + encoder_trend_hidden[1]
        decoder_hidden = (decoder_hidden, decoder_cell)

        outputs = []
        seq_len_out = self.seq_len_out
        teacher_forcing = self.is_teacher_forcing()
        for i in range(seq_len_out):
            if len(outputs) > 0:
                if teacher_forcing:
                    decoder_input = torch.cat(
                        (decoder_input[:, 1:, :], trg_teacher_forcing[:, len(outputs) - 1:len(outputs), :]), dim=1)
                else:
                    decoder_input = torch.cat((decoder_input[:, 1:, :], outputs[-1]), dim=1)
            else:
                decoder_input = decoder_input_seq
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            decoder_output = decoder_output[:, decoder_output.shape[1] - 1:, :]
            decoder_output = self.decoder_output_linear(decoder_output)
            outputs.append(decoder_output)

        # outputs: (batch_size, seq_length, output_size)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.view((outputs.size(0), outputs.size(1), outputs.size(3)))
        return outputs, trg_y
