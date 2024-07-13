from . import Linear
from . import Dlinear
from . import EncoderDecoderLSTM
from . import TimeSeriesTransformer
from . import MEAformer
from . import DMEAformer
from . import TCDformer
import sys
import torch


def get_model(model_params=None):
    DEVICE = model_params['DEVICE']
    seq_len_in = model_params['seq_len_in']
    seq_len_out = model_params['seq_len_out']
    seq_len_dec = model_params['seq_len_dec']
    num_features = model_params['num_features']
    num_outputs = model_params['num_outputs']
    teacher_forcing = model_params['teacher_forcing']

    model = None
    if model_params['model'] == 'linear':
        model = Linear(seq_len_in=seq_len_in, num_features=num_features, num_outputs=num_outputs,
                       seq_len_out=seq_len_out)

    elif model_params['model'] == 'dlinear':
        individual = model_params['individual']
        channel = model_params['channel']
        model = Dlinear(seq_len=seq_len_in, pred_len=seq_len_out,num_features=num_features, num_outputs=num_outputs, individual=individual, channel=channel)

    elif model_params['model'] == 'lstm':
        input_size, hidden_size, num_layers = num_features, model_params['hidden_size'], model_params['num_layers']
        output_size = num_outputs
        model = EncoderDecoderLSTM(input_size, hidden_size, output_size, num_layers, seq_len_in, seq_len_out,
                                   seq_len_dec, teacher_forcing)

    elif model_params['model'] == 'transformer':
        ## Model parameters
        dim_val = model_params[
            'dim_val']  # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
        n_heads = model_params[
            'n_heads']  # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
        n_encoder_layers = model_params[
            'n_encoder_layers']  # Number of times the encoder layer is stacked in the encoder
        n_decoder_layers = model_params[
            'n_decoder_layers']  # Number of times the decoder layer is stacked in the decoder
        input_size = num_features  # The number of input variables. 1 if univariate forecasting.
        dec_seq_len = seq_len_dec  # length of input given to decoder. Can have any integer value.
        enc_seq_len = seq_len_in  # length of input given to encoder. Can have any integer value.
        output_sequence_length = seq_len_out  # Length of the target sequence, i.e. how many time steps should your forecast cover
        max_seq_len = enc_seq_len  # What's the longest sequence the model will encounter? Used to make the positional encoder
        dim_feedforward_encoder = model_params['dim_val']
        dim_feedforward_decoder = model_params['dim_val']
        num_predicted_features = num_outputs

        model = TimeSeriesTransformer(
            dim_val=dim_val,
            input_size=input_size,
            enc_seq_len=enc_seq_len,
            dec_seq_len=dec_seq_len,
            # max_seq_len=max_seq_len,
            out_seq_len=output_sequence_length,
            n_decoder_layers=n_decoder_layers,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            batch_first=False,
            dim_feedforward_encoder=dim_feedforward_encoder,
            dim_feedforward_decoder=dim_feedforward_decoder,
            num_predicted_features=num_predicted_features,
            device=DEVICE,
            teacher_forcing=teacher_forcing
        )

    elif model_params['model'] == 'meaformer':
        n_encoder_layers = model_params['n_encoder_layers']
        n_decoder_layers = model_params['n_decoder_layers']
        S = model_params['S']
        hidden_size = model_params['hidden_size']
        model = MEAformer(seq_len_in, seq_len_out, S, n_encoder_layers, n_decoder_layers, hidden_size,num_features, num_outputs)

    elif model_params['model'] == 'dmeaformer':
        n_encoder_layers = model_params['n_encoder_layers']
        n_decoder_layers = model_params['n_decoder_layers']
        S = model_params['S']
        hidden_size = model_params['hidden_size']
        model = DMEAformer(seq_len_in, seq_len_out, S, n_encoder_layers, n_decoder_layers, hidden_size,num_features, num_outputs)

    elif model_params['model'] == 'tcdformer':
        version = model_params['version']
        n_encoder_layers = model_params['n_encoder_layers']
        n_decoder_layers = model_params['n_decoder_layers']
        moving_avg = model_params['moving_avg']
        d_model = model_params['d_model']
        dropout = model_params['dropout']
        modes = model_params['modes']
        activation = model_params['activation']
        window_size = model_params['window_size']
        hidden_size = model_params['hidden_size']

        model = TCDformer(version=version, seq_len=seq_len_in, label_len=seq_len_dec, pred_len=seq_len_out,
                           e_layers=n_encoder_layers, d_layers=n_decoder_layers, enc_in=num_features,dec_in=num_outputs, c_out=num_outputs,
                           moving_avg=moving_avg, d_model=d_model, dropout=dropout, modes=modes, activation=activation,
                           window_size=window_size, hidden_size=hidden_size)
    model.to(DEVICE)
    # device = torch.device("cuda:0")
    # model.to(device)
    # print('number of model params:', sum(p.numel() for p in model.parameters()))
    return model
