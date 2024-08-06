import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        # Encoder
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        _, (h_n, c_n) = self.encoder(x, (h_0, c_0))

        # Using the last hidden state as the context vector (latent space representation)
        latent = h_n[-1]

        # Decoder
        latent_repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(latent_repeated, (h_n, c_n))

        return out, latent

def get_model(input_size, hidden_size, num_layers=2):
    return LSTMAutoencoder(input_size, hidden_size, num_layers)
