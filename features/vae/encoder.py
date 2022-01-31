import torch

from torch import nn


class Encoder(nn.Module):

    def __init__(self, in_dimension, layer_1d, layer_2d, layer_3d, latent_dimension):
        """
        Fully Connected layers to encode molecule to latent space
        """
        super(Encoder, self).__init__()
        self.latent_dimension = latent_dimension

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.Sequential(
            nn.Linear(in_dimension, layer_1d),
            nn.BatchNorm1d(layer_1d),
            nn.LeakyReLU(),
            # nn.Linear(layer_1d, layer_2d),
            # nn.BatchNorm1d(layer_2d),
            # nn.LeakyReLU(),
            # nn.Linear(layer_2d, layer_3d),
            # nn.LeakyReLU()
        )

        # Latent space mean
        self.encode_mu = nn.Linear(layer_1d, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(layer_1d, latent_dimension)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        mu = self.encode_mu(h1)
        log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
