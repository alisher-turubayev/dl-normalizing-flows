# RealNVP implementation from https://github.com/fmu2/realNVP

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules_realnvp import (
    ChannelwiseAffineCoupling,
    CheckerboardAffineCoupling
)

class RealNVP(nn.Module):
    def __init__(self, channels, image_size, prior, hps):
        """Initializes a RealNVP.
        Args:
            channels: number of channels in the images.
            image_size: size of each image (a square of image_size*image_size).
            prior: prior distribution over latent space Z.
            hps: an object of class Hyperparameters() with model hyperparameters.
        """
        super(RealNVP, self).__init__()
        self.prior = prior
        self.channels = channels
        self.image_size = image_size

        chan = channels
        size = image_size
        dim = hps.base_dim

        # architecture for CIFAR-10 (down to 16 x 16 x C)
        # SCALE 1: 3 x 32 x 32
        self.s1_ckbd = self.checkerboard_combo(chan, dim, size, hps)
        self.s1_chan = self.channelwise_combo(chan*4, dim, hps)
        try:
            self.order_matrix_1 = self.order_matrix(chan).cuda()
        except AssertionError:
            self.order_matrix_1 = self.order_matrix(chan)
        chan *= 2
        size //= 2

        # SCALE 2: 6 x 16 x 16
        self.s2_ckbd = self.checkerboard_combo(chan, dim, size, hps, final=True)

    def checkerboard_combo(self, in_out_dim, mid_dim, size, hps, final=False):
        """Construct a combination of checkerboard coupling layers.
        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            final: True if at final scale, False otherwise.
        Returns:
            A combination of checkerboard coupling layers.
        """
        if final:
            return nn.ModuleList([
                CheckerboardAffineCoupling(in_out_dim, mid_dim, size, 1., hps),
                CheckerboardAffineCoupling(in_out_dim, mid_dim, size, 0., hps),
                CheckerboardAffineCoupling(in_out_dim, mid_dim, size, 1., hps),
                CheckerboardAffineCoupling(in_out_dim, mid_dim, size, 0., hps)])
        else:
            return nn.ModuleList([
                CheckerboardAffineCoupling(in_out_dim, mid_dim, size, 1., hps), 
                CheckerboardAffineCoupling(in_out_dim, mid_dim, size, 0., hps),
                CheckerboardAffineCoupling(in_out_dim, mid_dim, size, 1., hps)])
        
    def channelwise_combo(self, in_out_dim, mid_dim, hps):
        """Construct a combination of channelwise coupling layers.
        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
        Returns:
            A combination of channelwise coupling layers.
        """
        return nn.ModuleList([
                ChannelwiseAffineCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseAffineCoupling(in_out_dim, mid_dim, 1., hps),
                ChannelwiseAffineCoupling(in_out_dim, mid_dim, 0., hps)])

    def squeeze(self, x):
        """Squeezes a C x H x W tensor into a 4C x H/2 x W/2 tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x 4C x H/2 x W/2).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C*4, H//2, W//2)
        return x

    def undo_squeeze(self, x):
        """unsqueezes a C x H x W tensor into a C/4 x 2H x 2W tensor.
        (See Fig 3 in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
        Returns:
            the squeezed tensor (B x C/4 x 2H x 2W).
        """
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C//4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C//4, H*2, W*2)
        return x

    def order_matrix(self, channel):
        """Constructs a matrix that defines the ordering of variables
        when downscaling/upscaling is performed.
        Args:
          channel: number of features.
        Returns:
          a kernel for rearrange the variables.
        """
        weights = np.zeros((channel*4, channel, 2, 2))
        ordering = np.array([[[[1., 0.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [0., 1.]]],
                             [[[0., 1.],
                               [0., 0.]]],
                             [[[0., 0.],
                               [1., 0.]]]])
        for i in range(channel):
            s1 = slice(i, i+1)
            s2 = slice(4*i, 4*(i+1))
            weights[s2, s1, :, :] = ordering
        shuffle = np.array([4*i for i in range(channel)]
                         + [4*i+1 for i in range(channel)]
                         + [4*i+2 for i in range(channel)]
                         + [4*i+3 for i in range(channel)])
        weights = weights[shuffle, :, :, :].astype('float32')
        return torch.tensor(weights)

    def factor_out(self, x, order_matrix):
        """Downscales and factors out the bottom half of the tensor.
        (See Fig 4(b) in the real NVP paper.)
        Args:
            x: input tensor (B x C x H x W).
            order_matrix: a kernel that defines the ordering of variables.
        Returns:
            the top half for further transformation (B x 2C x H/2 x W/2)
            and the Gaussianized bottom half (B x 2C x H/2 x W/2).
        """
        x = F.conv2d(x, order_matrix, stride=2, padding=0)
        [_, C, _, _] = list(x.size())
        (on, off) = x.split(C//2, dim=1)
        return on, off

    def restore(self, on, off, order_matrix):
        """Merges variables and restores their ordering.
        (See Fig 4(b) in the real NVP paper.)
        Args:
            on: the active (transformed) variables (B x C x H x W).
            off: the inactive variables (B x C x H x W).
            order_matrix: a kernel that defines the ordering of variables.
        Returns:
            combined variables (B x 2C x H x W).
        """
        x = torch.cat((on, off), dim=1)
        return F.conv_transpose2d(x, order_matrix, stride=2, padding=0)

    def g(self, z):
        x, x_off_1 = self.factor_out(z, self.order_matrix_1)

        for i in reversed(range(len(self.s2_ckbd))):
            x, _ = self.s2_ckbd[i](x, reverse=True)

        x = self.restore(x, x_off_1, self.order_matrix_1)

        # SCALE 1: 32(64) x 32(64)
        x = self.squeeze(x)
        for i in reversed(range(len(self.s1_chan))):
            x, _ = self.s1_chan[i](x, reverse=True)
        x = self.undo_squeeze(x)

        for i in reversed(range(len(self.s1_ckbd))):
            x, _ = self.s1_ckbd[i](x, reverse=True)

        return x

    def f(self, x):
        z, log_diag_J = x, torch.zeros_like(x)

        # SCALE 1: 32(64) x 32(64)
        for i in range(len(self.s1_ckbd)):
            z, inc = self.s1_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
        for i in range(len(self.s1_chan)):
            z, inc = self.s1_chan[i](z)
            log_diag_J = log_diag_J + inc
        z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

        z, z_off_1 = self.factor_out(z, self.order_matrix_1)
        log_diag_J, log_diag_J_off_1 = self.factor_out(log_diag_J, self.order_matrix_1)

        # SCALE 2: 16(32) x 16(32)
        for i in range(len(self.s2_ckbd)):
            z, inc = self.s2_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        z = self.restore(z, z_off_1, self.order_matrix_1)
        log_diag_J = self.restore(log_diag_J, log_diag_J_off_1, self.order_matrix_1)

        return z, log_diag_J

    def log_prob(self, x):
        """Computes data log-likelihood.
        (See Eq(2) and Eq(3) in the real NVP paper.)
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_diag_J = self.f(x)
        log_det_J = torch.sum(log_diag_J, dim=(1, 2, 3))
        log_prior_prob = torch.sum(self.prior.log_prob(z), dim=(1, 2, 3))
        return log_prior_prob + log_det_J

    def sample(self, size):
        """Generates samples.
        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        C = self.channels
        H = W = self.image_size
        z = self.prior.sample((size, C, H, W))
        return self.g(z)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input and sum of squares of scaling factors.
            (the latter is used in L2 regularization.)
        """
        weight_scale = None
        for name, param in self.named_parameters():
            param_name = name.split('.')[-1]
            if param_name in ['weight_g', 'scale'] and param.requires_grad:
                if weight_scale is None:
                    weight_scale = torch.pow(param, 2).sum()
                else:
                    weight_scale = weight_scale + torch.pow(param, 2).sum()
        return self.log_prob(x), weight_scale