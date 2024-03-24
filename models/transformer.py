import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math
import random


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


class TimeSeriesTransformer(nn.Module):

    def __init__(self,
                 input_size: int,
                 enc_seq_len: int,
                 dec_seq_len: int,
                 batch_first: bool,
                 out_seq_len: int,
                 dim_val: int = 512,
                 n_encoder_layers: int = 4,
                 n_decoder_layers: int = 4,
                 n_heads: int = 8,
                 dropout_encoder: float = 0.2,
                 dropout_decoder: float = 0.2,
                 dropout_pos_enc: float = 0.1,
                 dim_feedforward_encoder: int = 2048,
                 dim_feedforward_decoder: int = 2048,
                 num_predicted_features: int = 1,
                 device=None,
                 teacher_forcing=0,
                 ):

        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__()
        self.name = 'transformer'
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len
        self.device = device
        self.counter = 0
        self.teacher_forcing = teacher_forcing
        self.epoch_portion = 0

        # print("input_size is: {}".format(input_size))
        # print("dim_val is: {}".format(dim_val))

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size,
            out_features=dim_val
        )

        self.decoder_input_layer = nn.Linear(
            in_features=num_predicted_features,
            out_features=dim_val
        )

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=num_predicted_features
        )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            batch_first=batch_first
        )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        # print( dim_val, n_heads,dim_feedforward_encoder,dropout_encoder,batch_first)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )
        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers,
            norm=None
        )

    def generate_masks(self, dec_seq_len=None):
        """
        Generates an upper-triangular matrix of -inf, with zeros on diag.
        Source:
        https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        Args:
            dim1: int, for both src and tgt masking, this must be target sequence
                length
            dim2: int, for src masking this must be encoder sequence length (i.e.
                the length of the input sequence to the model),
                and for tgt masking, this must be target sequence length
        Return:
            A Tensor of shape [dim1, dim2]
        """
        if dec_seq_len is None:
            dec_seq_len = self.dec_seq_len
        src_mask = torch.triu(torch.ones(dec_seq_len, self.enc_seq_len) * float('-inf'), diagonal=1)
        tgt_mask = torch.triu(torch.ones(dec_seq_len, dec_seq_len) * float('-inf'), diagonal=1)
        if self.device is not None:
            src_mask, tgt_mask = src_mask.to(self.device), tgt_mask.to(self.device)
        return src_mask, tgt_mask

    def is_teacher_forcing(self):

        if self.epoch_portion < self.teacher_forcing:
            return True
        return False

    def forward(self, src, trg, trg_y, trg_teacher_forcing, epoch_portion=0) -> Tensor:
        """
        Returns a tensor of shape:
        [target_sequence_length, batch_size, num_predicted_features]

        Args:
            src: the encoder's output sequence. Shape: (S,E) for unbatched input,
                 (S, N, E) if batch_first=False or (N, S, E) if
                 batch_first=True, where S is the source sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input,
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if
                 batch_first=True, where T is the target sequence length,
                 N is the batch size, and E is the number of features (1 if univariate)
            src_mask: the mask for the src sequence to prevent the model from
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """
        self.epoch_portion = epoch_portion

        src = src.view((src.size(1), src.size(0), src.size(2)))
        trg = trg.view((trg.size(1), trg.size(0), trg.size(2)))
        trg_teacher_forcing = trg_teacher_forcing.view(
            (trg_teacher_forcing.size(1), trg_teacher_forcing.size(0), trg_teacher_forcing.size(2)))

        src, tgt, trg_teacher_forcing = src, trg, trg_teacher_forcing
        self.counter += 1
        # print("From model.forward(): Size of src as given to forward(): {}".format(src.size()))
        # print("From model.forward(): tgt size = {}".format(tgt.size()))

        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(
            src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after input layer: {}".format(src.size()))

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(
            src)  # src shape: [batch_size, src length, dim_val] regardless of number of input features
        # print("From model.forward(): Size of src after pos_enc layer: {}".format(src.size()))

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length.
        # (https://github.com/huggingface/transformers/issues/4083)
        encoder_hidden = self.encoder(  # src shape: [batch_size, enc_seq_len, dim_val]
            src=src
        )
        # print("From model.forward(): Size of src after encoder: {}".format(src.size()))
        decoder_hidden = encoder_hidden

        teacher_forcing = self.is_teacher_forcing()
        if teacher_forcing:
            # Pass decoder input through decoder input layer
            decoder_input = torch.cat([tgt] + [trg_teacher_forcing[:-1, :, :]], dim=0)
            # decoder_input = trg_teacher_forcing
            decoder_output = self.decoder_input_layer(
                decoder_input)  # src shape: [target sequence length, batch_size, dim_val] regardless of number of input features
            # print("From model.forward(): Size of decoder_output aftetrgr linear decoder layer: {}".format(decoder_output.size()))

            # if src_mask is not None:
            # print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
            # if tgt_mask is not None:
            # print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))
            src_mask, tgt_mask = self.generate_masks(decoder_output.shape[0])

            # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
            decoder_output = self.decoder(
                tgt=decoder_output,
                memory=decoder_hidden,
                tgt_mask=tgt_mask,
                memory_mask=src_mask
            )

            # print("From model.forward(): decoder_output shape after decoder: {}".format(decoder_output.shape))

            # Pass through linear mapping
            decoder_output = self.linear_mapping(decoder_output)  # shape [batch_size, target seq len]
            # print("From model.forward(): decoder_output size after linear_mapping = {}".format(decoder_output.size()))
            decoder_output_trg = decoder_output[decoder_output.shape[0] - trg_teacher_forcing.shape[0]:, :, :]
            # print("teacher_forcing",teacher_forcing,"decoder_output=",decoder_output.shape,"tgt=",tgt.shape,"trg_teacher_forcing=",trg_teacher_forcing.shape,
            #   "src=",src.shape,"decoder_input=",decoder_input.shape,"decoder_output_trg=",decoder_output_trg.shape )

        else:
            outputs = []
            for i in range(self.out_seq_len):

                decoder_input = torch.cat([tgt] + outputs, dim=0)
                decoder_input = decoder_input.detach()
                decoder_output = self.decoder_input_layer(decoder_input)
                src_mask, tgt_mask = (self.generate_masks
                                      (decoder_output.shape[0]))

                decoder_output = self.decoder(
                    tgt=decoder_output,
                    memory=decoder_hidden,
                    tgt_mask=tgt_mask,
                    memory_mask=src_mask
                )

                decoder_output = decoder_output[decoder_output.shape[0] - 1:, :, :]
                decoder_output = self.linear_mapping(decoder_output)
                if i < (self.out_seq_len - 1):
                    decoder_output = decoder_output.detach()
                outputs.append(decoder_output)

            decoder_output_trg = torch.cat(outputs, dim=0)
            # print("teacher_forcing","outputs",len(outputs),teacher_forcing,"decoder_output_trg=",decoder_output_trg.shape,"src=",src.shape,"decoder_input=",decoder_input.shape )

        pred = decoder_output_trg
        pred = pred.view((pred.size(1), pred.size(0), pred.size(2)))

        return pred, trg_y