import torch
from torch import nn, Tensor
import random
import math

class PositionalEncoder(nn.Module):
    def __init__(
            self,
            dropout: float = 0.1,
            max_seq_len: int = 5000,
            d_model: int = 512,
            batch_first: bool = False
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, 1, d_model)

        pe[:, 0, 0::2] = torch.sin(position * div_term)

        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        x_pe = self.pe[:x.size(self.x_dim)]
        if self.batch_first:
            x_pe = x_pe.view((x_pe.size(1), x_pe.size(0), x_pe.size(2)))

        # print(x.shape,self.pe.shape, x_pe.shape)
        x = x + x_pe

        return self.dropout(x)


class AttentionLSTM(nn.Module):
    def __init__(self, input_size,
                 hidden_size,
                 output_size,
                 num_layers,
                 seq_len_in,
                 seq_len_out,
                 seq_len_dec,
                 dim_val,
                 n_encoder_layers,
                 n_heads,
                 batch_first,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dropout_encoder: float = 0.2,
                 teacher_forcing=0):

        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = input_size
        self.output_size = output_size

        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.seq_len_dec = seq_len_dec
        self.teacher_forcing = teacher_forcing

        self.counter = 0
        self.epoch_portion = 0

        # self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
        #                             batch_first=True, bidirectional=True)

        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            batch_first=batch_first
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        self.decoder_lstm = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=2 * num_layers,
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

        # print("dfffffffffffffffffffffffffff: ", trg.shape)
        src = src.view((src.size(1), src.size(0), src.size(2)))
        # trg = trg.view((trg.size(1), trg.size(0), trg.size(2)))
        # trg_teacher_forcing = trg_teacher_forcing.view(
        #     (trg_teacher_forcing.size(1), trg_teacher_forcing.size(0), trg_teacher_forcing.size(2)))

        src, decoder_input_seq, trg_teacher_forcing = src, trg, trg_teacher_forcing
        self.counter += 1

        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        encoder_output = self.encoder(src=src)   # src shape: [batch_size, enc_seq_len, dim_val]
        encoder_output = encoder_output.permute(1, 0, 2)
        encoder_final_state = encoder_output[:, -1, :]
        # print("encoder output shape:", encoder_hidden.shape)
        # encoder_input_seq, decoder_input_seq, trg_teacher_forcing = src, trg, trg_teacher_forcing
        # self.counter += 1
        # # input_seq: (batch_size, seq_length, input_size)
        # # Encoding phase
        # encoder_outputs, encoder_hidden = self.encoder_lstm(encoder_input_seq)
        #
        # decoder_input = decoder_input_seq
        hidden = encoder_final_state.unsqueeze(0).repeat(self.decoder_lstm.num_layers, 1, 1)  # shape: (num_lstm_layers, batch_size, lstm_hidden_dim)
        cell = torch.zeros(self.decoder_lstm.num_layers, encoder_output.size(0), self.decoder_lstm.hidden_size).to(
            encoder_output.device)  # shape: (num_lstm_layers, batch_size, lstm_hidden_dim)
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
            decoder_output, decoder_hidden = self.decoder_lstm(decoder_input, (hidden, cell))
            decoder_output = decoder_output[:, decoder_output.shape[1] - 1:, :]
            decoder_output = self.decoder_output_linear(decoder_output)
            outputs.append(decoder_output)

        # outputs: (batch_size, seq_length, output_size)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.view((outputs.size(0), outputs.size(1), outputs.size(3)))
        return outputs, trg_y
