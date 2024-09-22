import torch
import torch.nn as nn
import random


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x) # 提取的趋势
        res = x - moving_mean # 季节项
        return res, moving_mean


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
        # input_seq: (batch_size, seq_length, input_size)
        # Encoding phase
        encoder_seasonal_outputs, encoder_seasonal_hidden = self.encoder_seasonal_lstm(encoder_input_seasonal)
        encoder_trend_outputs, encoder_trend_hidden = self.encoder_trend_lstm(encoder_input_trend)
        # print("seasonal0: ", encoder_seasonal_hidden[0].shape)
        # print("seasonal1: ", encoder_seasonal_hidden[1].shape)
        # print("trend0: ", encoder_trend_hidden[0].shape)
        # print("trend1: ", encoder_trend_hidden[1].shape)

        decoder_input = decoder_input_seq
        decoder_hidden = encoder_seasonal_hidden[0] + encoder_trend_hidden[0]
        decoder_cell = encoder_seasonal_hidden[1] + encoder_trend_hidden[1]
        decoder_hidden = (decoder_hidden, decoder_cell)  # Use last encoder hidden state as decoder's initial hidden state
        # print("hidden0: ", decoder_hidden[0].shape)
        # print("hidden1: ", decoder_hidden[1].shape)

        outputs = []

        seq_len_out = self.seq_len_out
        teacher_forcing = self.is_teacher_forcing()
        for i in range(seq_len_out):
            # print("shape decoder_input",i,decoder_input.shape)
            if len(outputs) > 0:
                if teacher_forcing:
                    # decoder_input = torch.cat([decoder_input_seq] + [trg_teacher_forcing[:,:len(outputs),:]], dim=1)
                    decoder_input = trg_teacher_forcing[:, len(outputs) - 1:len(outputs), :]
                else:
                    # decoder_input = torch.cat([decoder_input_seq] + outputs, dim=1)
                    decoder_input = outputs[-1]
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
