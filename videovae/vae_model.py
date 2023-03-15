import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEs(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        #encoder
        self.encoder = Encoder(args.h_dim, args.downsample)

        self.fc_mu_1 = nn.Linear(args.h_dim * args.Linear_multi, args.h_dim)
        self.fc_mu_2 = nn.Linear(args.h_dim, args.z_dim)
        self.fc_var_1 = nn.Linear(args.h_dim * args.Linear_multi, args.h_dim)
        self.fc_var_2 = nn.Linear(args.h_dim, args.z_dim)

        # decoder
        self.decoder = Decoder(args.h_dim, args.downsample)

        self.dec_1 = nn.Linear(args.z_dim, args.h_dim)
        self.dec_2 = nn.Linear(args.h_dim, args.h_dim * args.Linear_multi)

        # self.pre_conv_mu = SamePadConv3d(args.h_dim, args.z_dim, 1)
        # self.pre_conv_sigma = SamePadConv3d(args.h_dim, args.z_dim, 1)
        # self.post_conv = SamePadConv3d(args.z_dim, args.h_dim, 1)

    def latent_shape(self):
        input_shape = (self.args.sequence_length, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])
    def encode(self, x):
        # q_phi(z|x)
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        # (16, 512 * 2 * 8 * 8)
        mu, log_sigma = self.fc_mu_2(self.fc_mu_1(h)), self.fc_var_2(self.fc_var_1(h))
        # flatten -->mu without sigma
        # mu, log_sigma = self.pre_conv(self.encoder(x)), self.pre_conv(self.encoder(x))
        return mu, log_sigma

    def decode(self, z):
        #p_zeta(x|z)
        h = self.dec_2(self.dec_1(z))
        h = h.view(-1, 512, 2, 8, 8)
        x = self.decoder(h)
        # x = self.decoder(self.post_conv(z))
        return x

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        sigma = log_sigma.exp()
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        # print(z_reparametrized.shape)
        x_reconstructed = self.decode(z_reparametrized)
        # print(x.shape, x_reconstructed.shape)
        return x_reconstructed, mu, log_sigma

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_channels', type=int, default=3)
        parser.add_argument('--h_dim', type=int, default=512)
        parser.add_argument('--Linear_multi', type=int, default=128)
        parser.add_argument('--z_dim', type=int, default=128)
        parser.add_argument('--downsample', nargs='+', type=int, default=(8, 8, 8))
        return parser

class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.convs = nn.ModuleList()
        max_ds = n_times_downsample.max()
        for i in range(max_ds):
            in_channels = 3 if i == 0 else n_hiddens
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            conv = SamePadConv3d(in_channels, n_hiddens, 4, stride=stride)
            self.convs.append(conv)
            n_times_downsample -= 1
        self.conv_last = SamePadConv3d(in_channels, n_hiddens, kernel_size=3)

        self.batchnorm = nn.BatchNorm3d(n_hiddens)
    def forward(self, x):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h))
        h = self.conv_last(h)
        h = self.batchnorm(h)
        return h

class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample):
        super().__init__()
        self.batchnorm = nn.BatchNorm3d(n_hiddens)
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        self.convts = nn.ModuleList()
        for i in range(max_us):
            out_channels = 3 if i == max_us - 1 else n_hiddens
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            convt = SamePadConvTranspose3d(n_hiddens, out_channels, 4,
                                           stride=us)
            self.convts.append(convt)
            n_times_upsample -= 1

    def forward(self, x):
        h = self.batchnorm(x)
        for i, convt in enumerate(self.convts):
            h = convt(h)
            if i < len(self.convts) - 1:
                h = F.relu(h)
        return h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input))
