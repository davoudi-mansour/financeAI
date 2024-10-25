import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len_in, seq_len_out):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = input_size
        self.output_size = output_size

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size*seq_len_out)

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0):
        self.counter += 1
        output, hidden = self.lstm(src)
        outputs = output[:, output.shape[1] - 1:, :]
        outputs = self.linear(outputs)

        # outputs: (batch_size, seq_length, output_size)
        outputs = outputs.view(outputs.size(0), -1, self.output_size)
        return outputs, trg_y
