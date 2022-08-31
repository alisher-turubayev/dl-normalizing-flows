# Utilizing Normalizing Flows for Anime Face Generation
# 
# Deep Learning Summer 2022 - Final Project
# Hasso-Plattner Institute
# 
# Code adapted by Alisher Turubayev, M.Sc. in Digital Health Student
# 
# References to algorithms:
#   https://arxiv.org/pdf/1605.08803.pdf - RealNVP
#   https://arxiv.org/pdf/1511.06434.pdf - DCGAN
# 
# Code references:
#   https://github.com/ikostrikov/pytorch-flows/,
#   https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py,
#   https://github.com/fmu2/realNVP
# 
# All code utilitzed in this project is a property of the respective authors. Code was used in good faith
#   for learning purposes and for the completion of the final project. The author of this notice does not 
#   claim any rights of ownership and/or originality.
# 
# Code by Ilya Kostrikov (ikostrikov) and Fangzhou Mu (fmu2) is licensed under MIT License. 
#   Code by Nathan Inkawhich (inkawich) is licensed under BSD 3-Clause License. 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules_realnvp import (
    ChannelwiseAffineCoupling,
    CheckerboardAffineCoupling
)

# RealNVP implementation by @fmu2 - https://github.com/fmu2/realNVP
class RealNVP(nn.Module):
    def __init__(self, channels, image_size, prior, hps):
        super(RealNVP, self).__init__()
        self.prior = prior
        self.channels = channels
        self.image_size = image_size

        chan = channels
        size = image_size
        dim = hps.base_dim

        # The following is the recursive (conceptually) scaling of spatial size 
        #   Originally, two scaling operations were performed (3 x 32 x 32 -> 6 x 16 x 16 -> 12 x 4 x 4);
        #   however, a larger input image size necessitated a change to 5 scaling operations.
        # This approach is described in section 3.6 of the RealNVP paper
        # Scale 1: 3 x 64 x 64
        self.s1_ckbd = self.checkerboard_combo(chan, dim, size, hps)
        self.s1_chan = self.channelwise_combo(chan*4, dim*2, hps)
        try:
            self.order_matrix_1 = self.order_matrix(chan).cuda()
        except AssertionError:
            self.order_matrix_1 = self.order_matrix(chan)
        chan *= 2
        size //= 2
        dim *= 2

        # Scale 2: 6 x 32 x 32
        self.s2_ckbd = self.checkerboard_combo(chan, dim, size, hps)
        self.s2_chan = self.channelwise_combo(chan*4, dim*2, hps)
        try:
            self.order_matrix_2 = self.order_matrix(chan).cuda()
        except AssertionError:
            self.order_matrix_2 = self.order_matrix(chan)
        chan *= 2
        size //= 2
        dim *= 2

        # Scale 3: 12 x 16 x 16
        self.s3_ckbd = self.checkerboard_combo(chan, dim, size, hps)
        self.s3_chan = self.channelwise_combo(chan*4, dim*2, hps)
        try:
            self.order_matrix_3 = self.order_matrix(chan).cuda()
        except AssertionError:
            self.order_matrix_3 = self.order_matrix(chan)
        chan *= 2
        size //= 2
        dim *= 2

        # Scale 4: 24 x 8 x 8
        self.s4_ckbd = self.checkerboard_combo(chan, dim, size, hps)
        self.s4_chan = self.channelwise_combo(chan*4, dim*2, hps)
        try:
            self.order_matrix_4 = self.order_matrix(chan).cuda()
        except AssertionError:
            self.order_matrix_4 = self.order_matrix(chan)
        chan *= 2
        size //= 2
        dim *= 2

        # Scale 5 (final): 48 x 4 x 4
        self.s5_ckbd = self.checkerboard_combo(chan, dim, size, hps, final=True)

    # Generates a combination of checkerboard affine coupling layers according to section 3.2 of the RealNVP paper
    def checkerboard_combo(self, in_out_dim, mid_dim, size, hps, final=False):
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
        
    # Generates a combination of channelwise affine coupling layers according to section 3.2 of the RealNVP paper
    def channelwise_combo(self, in_out_dim, mid_dim, hps):
        return nn.ModuleList([
                ChannelwiseAffineCoupling(in_out_dim, mid_dim, 0., hps),
                ChannelwiseAffineCoupling(in_out_dim, mid_dim, 1., hps),
                ChannelwiseAffineCoupling(in_out_dim, mid_dim, 0., hps)])

    # Squeezing operation described in section 3.6 of the RealNVP paper
    #   This operation transforms the input tensor from size C x H x W to C * 4 x H / 2 x W / 2 tensor
    #   allowing for a trade between spatial size and number of channels
    def squeeze(self, x):
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C*4, H//2, W//2)
        return x

    # Reverses the squeeze operation
    #   This operation is described in section 3.6 of the RealNVP paper
    def undo_squeeze(self, x):
        [B, C, H, W] = list(x.size())
        x = x.reshape(B, C//4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C//4, H*2, W*2)
        return x

    # This is a supporting function not defined in the original paper
    # As such, the origninal comment was left as is
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

    # Transformation function from Z to X
    def g(self, z):
        x, x_off_1 = self.factor_out(z, self.order_matrix_1)
        x, x_off_2 = self.factor_out(x, self.order_matrix_2)
        x, x_off_3 = self.factor_out(x, self.order_matrix_3)
        x, x_off_4 = self.factor_out(x, self.order_matrix_4)

        for i in reversed(range(len(self.s5_ckbd))):
            x, _ = self.s5_ckbd[i](x, reverse=True)
        
        x = self.restore(x, x_off_4, self.order_matrix_4)

        # Scale 4: 8 x 8
        x = self.squeeze(x)
        for i in reversed(range(len(self.s4_chan))):
            x, _ = self.s4_chan[i](x, reverse=True)
        x = self.undo_squeeze(x)

        for i in reversed(range(len(self.s4_ckbd))):
            x, _ = self.s4_ckbd[i](x, reverse=True)

        x = self.restore(x, x_off_3, self.order_matrix_3)

        # Scale 3: 8(16) x 8(16)
        x = self.squeeze(x)
        for i in reversed(range(len(self.s3_chan))):
            x, _ = self.s3_chan[i](x, reverse=True)
        x = self.undo_squeeze(x)

        for i in reversed(range(len(self.s3_ckbd))):
            x, _ = self.s3_ckbd[i](x, reverse=True)

        x = self.restore(x, x_off_2, self.order_matrix_2)

        # Scale 2: 16(32) x 16(32)
        x = self.squeeze(x)
        for i in reversed(range(len(self.s2_chan))):
            x, _ = self.s2_chan[i](x, reverse=True)
        x = self.undo_squeeze(x)

        for i in reversed(range(len(self.s2_ckbd))):
            x, _ = self.s2_ckbd[i](x, reverse=True)

        x = self.restore(x, x_off_1, self.order_matrix_1)

        # Scale 1: 32(64) x 32(64)
        x = self.squeeze(x)
        for i in reversed(range(len(self.s1_chan))):
            x, _ = self.s1_chan[i](x, reverse=True)
        x = self.undo_squeeze(x)

        for i in reversed(range(len(self.s1_ckbd))):
            x, _ = self.s1_ckbd[i](x, reverse=True)

        return x

    # Transformation function X to Z
    def f(self, x):
        z, log_diag_J = x, torch.zeros_like(x)

        # Scale 1: 32(64) x 32(64)
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

        # Scale 2: 16(32) x 16(32)
        for i in range(len(self.s2_ckbd)):
            z, inc = self.s2_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
        for i in range(len(self.s2_chan)):
            z, inc = self.s2_chan[i](z)
            log_diag_J = log_diag_J + inc
        z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

        z, z_off_2 = self.factor_out(z, self.order_matrix_2)
        log_diag_J, log_diag_J_off_2 = self.factor_out(log_diag_J, self.order_matrix_2)

        # Scale 3: 8(16) x 8(16)
        for i in range(len(self.s3_ckbd)):
            z, inc = self.s3_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
        for i in range(len(self.s3_chan)):
            z, inc = self.s3_chan[i](z)
            log_diag_J = log_diag_J + inc
        z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

        z, z_off_3 = self.factor_out(z, self.order_matrix_3)
        log_diag_J, log_diag_J_off_3 = self.factor_out(log_diag_J, self.order_matrix_3)

        # Scale 4: 4(8) x 4(8)
        for i in range(len(self.s4_ckbd)):
            z, inc = self.s4_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        z, log_diag_J = self.squeeze(z), self.squeeze(log_diag_J)
        for i in range(len(self.s4_chan)):
            z, inc = self.s4_chan[i](z)
            log_diag_J = log_diag_J + inc
        z, log_diag_J = self.undo_squeeze(z), self.undo_squeeze(log_diag_J)

        z, z_off_4 = self.factor_out(z, self.order_matrix_4)
        log_diag_J, log_diag_J_off_4 = self.factor_out(log_diag_J, self.order_matrix_4)

        # Scale 5: 4 x 4
        for i in range(len(self.s5_ckbd)):
            z, inc = self.s5_ckbd[i](z)
            log_diag_J = log_diag_J + inc

        z = self.restore(z, z_off_4, self.order_matrix_4)
        log_diag_J = self.restore(log_diag_J, log_diag_J_off_4, self.order_matrix_4)

        z = self.restore(z, z_off_3, self.order_matrix_3)
        z = self.restore(z, z_off_2, self.order_matrix_2)
        log_diag_J = self.restore(log_diag_J, log_diag_J_off_3, self.order_matrix_3)
        log_diag_J = self.restore(log_diag_J, log_diag_J_off_2, self.order_matrix_2)

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