import torch
import torch.nn as nn
import math
from encoder import Encoder
from decoder import Decoder


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, std = self.encoder(input)

        e = torch.zeros(mean.shape).normal_()
        z = std * e + mean

        y = self.decoder(z)

        eps = 1e-8
        L_reconstruction = input * y.log() + (1 - input) * (1 - y).log()
        KLD = 0.5 * (std.pow(2) + mean.pow(2) - 1 - torch.log(std.pow(2)+eps))

        elbo = KLD.sum(dim=-1) - L_reconstruction.sum(dim=-1)
        average_negative_elbo = elbo.mean()

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        samples = torch.randn((n_samples, self.z_dim))
        y = self.decoder(samples)

        im_means = y.reshape(n_samples, 1, 28, 28)
        sampled_ims = torch.bernoulli(im_means)

        return sampled_ims, im_means

    def manifold_sample(self, n_samples):
        n = int(math.sqrt(n_samples))
        xy = torch.zeros(n_samples, 2)
        xy[:, 0] = torch.arange(0.01, n, 1 / n) % 1
        xy[:, 1] = (torch.arange(0.01, n_samples, 1) / n).float() / n
        z = torch.erfinv(2 * xy - 1) * math.sqrt(2)

        with torch.no_grad():
            mean = self.decoder(z)

        return mean