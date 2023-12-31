from math import floor

import torch
import torch.nn as nn

# inspired by
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py


class AutoEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_channels=3,
        encoder_channel_out_sizes=[16, 32, 64, 128, 256],
        input_image_h_w=(218, 178),
    ):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_channel_out_sizes = encoder_channel_out_sizes
        self.encoder_out_channels = encoder_channel_out_sizes[-1]
        # Need Define a linear layer
        # Can atuomatically determine the required size bsed on the input
        # size and the properties of the conv layers
        self.conv_out_shape = self.recursive_apply(
            input_image_h_w, self.conv_output_shape, len(encoder_channel_out_sizes)
        )
        self.linear_layer_size = (
            self.conv_out_shape[0] * self.conv_out_shape[1] * self.encoder_out_channels
        )

        self._init_encoder(input_channels, encoder_channel_out_sizes)
        self._init_decoder(input_channels)

    def _init_encoder(self, input_channels, encoder_channel_out_sizes):
        # initialise the encoder layers
        encoder_input_sizes = [input_channels] + encoder_channel_out_sizes[:-1]
        modules = []
        for in_size, out_size in zip(encoder_input_sizes, encoder_channel_out_sizes):
            modules.append(self._get_cnn_module(in_size, out_size))

        self.encoder_linear_mean = nn.Linear(self.linear_layer_size, self.latent_dim)

        self.encoder = nn.Sequential(
            *modules,
            nn.Flatten(),
        )  # nn.BatchNorm1d(1))

    def _init_decoder(self, input_channels):
        self.decoder_linear = nn.Linear(self.latent_dim, self.linear_layer_size)

        modules = []

        output_channel_sizes = self.encoder_channel_out_sizes[::-1][1:] + [
            input_channels
        ]

        # output_padding = [(1, 0), 0, 0, 1]
        output_padding = [(1, 1), (1, 0), 0, 0, 1]
        for in_size, out_size, padding in zip(
            self.encoder_channel_out_sizes[::-1], output_channel_sizes, output_padding
        ):
            modules.append(
                self._get_transpose_cnn_module(in_size, out_size, out_padding=padding)
            )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: torch.Tensor):
        """Helper function making it easier to encode
        images
        """
        return self.encoder_linear_mean(self.encoder(input))

    def decode(self, input: torch.Tensor):
        input = self.decoder_linear(input)
        out = input.view(-1, self.encoder_out_channels, *self.conv_out_shape)
        return self.decoder(out)

    def forward(self, input):
        encoded = self.encode(input)
        return self.decode(encoded)

    @staticmethod
    def recursive_apply(input, func, n):
        for i in range(n):
            input = func(input)
        return input

    @staticmethod
    def conv_output_shape(h_w, kernel_size=3, stride=2, pad=1, dilation=1):
        """
        Calculate the height and width of conv2D output given input params
        """

        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor(
            ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1
        )
        w = floor(
            ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
        )
        return h, w

    @staticmethod
    def _get_cnn_module(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        """Helper function that creates a sequence of conv2d with bath normalization
        and an activation function
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    @staticmethod
    def _get_transpose_cnn_module(
        in_channels, out_channels, kernel_size=3, stride=2, padding=1, out_padding=0
    ):
        # https://stackoverflow.com/questions/60700472/pytorch-autoencoder-decoded-output-dimension-not-the-same-as-input

        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=out_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )


class VAE(AutoEncoder):
    """construct a VAE by using the basic same skeleton
    as the AutoEncoder, but by adding and overwriting a few things
    """

    def __init__(
        self,
        latent_dim,
        input_channels=3,
        encoder_channel_out_sizes=[16, 32, 64, 128, 256],
        input_image_h_w=(218, 178),
    ):
        super().__init__(
            latent_dim,
            input_channels,
            encoder_channel_out_sizes,
            input_image_h_w,
        )
        # Add another linear layer for the log variance
        self.encoder_linear_logvar = nn.Linear(self.linear_layer_size, self.latent_dim)

    @staticmethod
    def reparameterise(mu, logvar):
        """Implement the reparameterisation trick"""
        # scale the log variance to std
        std = torch.exp(0.5 * logvar)
        # sample from a standard noem
        normal_noise = torch.randn_like(std)
        return mu + (std * normal_noise)

    def encode(self, input: torch.Tensor):
        """Helper function making it easier to encode
        images
        """
        x = self.encoder(input)
        mu = self.encoder_linear_mean(x)
        logvar = self.encoder_linear_logvar(x)
        return self.reparameterise(mu, logvar), mu, logvar

    def forward(self, input):
        encoded, mu, logvar = self.encode(input)
        return self.decode(encoded), mu, logvar

    def loss_function(self, input, target, mu, logvar):
        # summed over each element in the latent space
        # the mean is taken over training examples
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0
        )
        kld_loss = kld_loss * 1e-5

        reconstruction_loss = nn.functional.mse_loss(input, target)

        return (
            reconstruction_loss + kld_loss,
            reconstruction_loss.detach(),
            kld_loss.detach(),
        )


# https://arxiv.org/pdf/1606.02228.pdf
