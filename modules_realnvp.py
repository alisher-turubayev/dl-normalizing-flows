### RealNVP modules from https://github.com/fmu2/realNVP
import numpy as np

import torch
import torch.nn as nn

class WeightNormConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, 
        bias=True, weight_norm=True, scale=False):
        """Intializes a Conv2d augmented with weight normalization.
        (See torch.nn.utils.weight_norm for detail.)
        Args:
            in_dim: number of input channels.
            out_dim: number of output channels.
            kernel_size: size of convolving kernel.
            stride: stride of convolution.
            padding: zero-padding added to both sides of input.
            bias: True if include learnable bias parameters, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            scale: True if include magnitude parameters, False otherwise.
        """
        super(WeightNormConv2d, self).__init__()

        if weight_norm:
            self.conv = nn.utils.weight_norm(
                nn.Conv2d(in_dim, out_dim, kernel_size, 
                    stride=stride, padding=padding, bias=bias))
            if not scale:
                self.conv.weight_g.data = torch.ones_like(self.conv.weight_g.data)
                self.conv.weight_g.requires_grad = False    # freeze scaling
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, 
                stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, bottleneck, weight_norm):
        """Initializes a ResidualBlock.
        Args:
            dim: number of input and output features.
            bottleneck: True if use bottleneck, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualBlock, self).__init__()
        
        self.in_block = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU())
        if bottleneck:
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (1, 1), stride=1, padding=0, 
                    bias=True, weight_norm=weight_norm, scale=True))
        else:
            self.res_block = nn.Sequential(
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=False, weight_norm=weight_norm, scale=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                    bias=True, weight_norm=weight_norm, scale=True))

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        return x + self.res_block(self.in_block(x))

class ResidualModule(nn.Module):
    def __init__(self, in_dim, dim, out_dim, 
        res_blocks, bottleneck, skip, weight_norm):
        """Initializes a ResidualModule.
        Args:
            in_dim: number of input features.
            dim: number of features in residual blocks.
            out_dim: number of output features.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
        """
        super(ResidualModule, self).__init__()
        self.res_blocks = res_blocks
        self.skip = skip
        
        if res_blocks > 0:
            self.in_block = WeightNormConv2d(in_dim, dim, (3, 3), stride=1, 
                padding=1, bias=True, weight_norm=weight_norm, scale=False)
            self.core_block = nn.ModuleList(
                [ResidualBlock(dim, bottleneck, weight_norm) 
                for _ in range(res_blocks)])
            self.out_block = nn.Sequential(
                nn.BatchNorm2d(dim),
                nn.ReLU(),
                WeightNormConv2d(dim, out_dim, (1, 1), stride=1, padding=0, 
                    bias=True, weight_norm=weight_norm, scale=True))
        
            if skip:
                self.in_skip = WeightNormConv2d(dim, dim, (1, 1), stride=1, 
                    padding=0, bias=True, weight_norm=weight_norm, scale=True)
                self.core_skips = nn.ModuleList(
                    [WeightNormConv2d(
                        dim, dim, (1, 1), stride=1, padding=0, bias=True, 
                        weight_norm=weight_norm, scale=True) 
                    for _ in range(res_blocks)])
        else:
            if bottleneck:
                self.block = nn.Sequential(
                    WeightNormConv2d(in_dim, dim, (1, 1), stride=1, padding=0, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, dim, (3, 3), stride=1, padding=1, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, out_dim, (1, 1), stride=1, padding=0, 
                        bias=True, weight_norm=weight_norm, scale=True))
            else:
                self.block = nn.Sequential(
                    WeightNormConv2d(in_dim, dim, (3, 3), stride=1, padding=1, 
                        bias=False, weight_norm=weight_norm, scale=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                    WeightNormConv2d(dim, out_dim, (3, 3), stride=1, padding=1, 
                        bias=True, weight_norm=weight_norm, scale=True))

    def forward(self, x):
        """Forward pass.
        Args:
            x: input tensor.
        Returns:
            transformed tensor.
        """
        if self.res_blocks > 0:
            x = self.in_block(x)
            if self.skip:
                out = self.in_skip(x)
            for i in range(len(self.core_block)):
                x = self.core_block[i](x)
                if self.skip:
                    out = out + self.core_skips[i](x)
            if self.skip:
                x = out
            return self.out_block(x)
        else:
            return self.block(x)

class AbstractCoupling(nn.Module):
    def __init__(self, mask_config, hps):
        """Initializes an AbstractCoupling.
        Args:
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super(AbstractCoupling, self).__init__()
        self.mask_config = mask_config
        self.res_blocks = hps.res_blocks
        self.bottleneck = hps.bottleneck
        self.skip = hps.skip
        self.weight_norm = hps.weight_norm
        self.coupling_bn = hps.coupling_bn

    def build_mask(self, size, config=1.):
        """Builds a binary checkerboard mask.
        (Only for constructing masks for checkerboard coupling layers.)
        Args:
            size: height/width of features.
            config: mask configuration that determines which pixels to mask up.
                    if 1:        if 0:
                        1 0         0 1
                        0 1         1 0
        Returns:
            a binary mask (1: pixel on, 0: pixel off).
        """
        mask = np.arange(size).reshape(-1, 1) + np.arange(size)
        mask = np.mod(config + mask, 2)
        mask = mask.reshape(-1, 1, size, size)
        return torch.tensor(mask.astype('float32'))

    def batch_stat(self, x):
        """Compute (spatial) batch statistics.
        Args:
            x: input minibatch.
        Returns:
            batch mean and variance.
        """
        mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        var = torch.mean((x - mean)**2, dim=(0, 2, 3), keepdim=True)
        return mean, var

class CheckerboardAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, size, mask_config, hps):
        """Initializes a CheckerboardAffineCoupling.
        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            size: height/width of features.
            mask_config: mask configuration (see build_mask() for more detail).
            hps: the set of hyperparameters.
        """
        super(CheckerboardAffineCoupling, self).__init__(mask_config, hps)

        try:
            self.mask = self.build_mask(size, config = mask_config).cuda()
        except AssertionError:
            self.mask = self.build_mask(size, config = mask_config)
        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.in_bn = nn.BatchNorm2d(in_out_dim)
        self.block = nn.Sequential(        # 1st half of resnet: shift
            nn.ReLU(),                    # 2nd half of resnet: log_rescale
            ResidualModule(2*in_out_dim+1, mid_dim, 2*in_out_dim, 
                self.res_blocks, self.bottleneck, self.skip, self.weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [B, C, _, _] = list(x.size())
        mask = self.mask.repeat(B, 1, 1, 1)
        x_ = self.in_bn(x * mask)
        x_ = torch.cat((x_, -x_), dim=1)
        x_ = torch.cat((x_, mask), dim=1)    # 2C+1 channels
        (shift, log_rescale) = self.block(x_).split(C, dim=1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        shift = shift * (1. - mask)
        log_rescale = log_rescale * (1. - mask)
        
        log_diag_J = log_rescale     # See Eq(6) in real NVP 
        # See Eq(7) and Eq(8) and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
                var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = x * torch.exp(0.5 * torch.log(var + 1e-5) * (1. - mask)) \
                    + mean * (1. - mask)
            x = (x - shift) * torch.exp(-log_rescale)
        else:
            x = x * torch.exp(log_rescale) + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(x)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                x = self.out_bn(x) * (1. - mask) + x * mask
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5) * (1. - mask)
        return x, log_diag_J

class ChannelwiseAffineCoupling(AbstractCoupling):
    def __init__(self, in_out_dim, mid_dim, mask_config, hps):
        """Initializes a ChannelwiseAffineCoupling.
        Args:
            in_out_dim: number of input and output features.
            mid_dim: number of features in residual blocks.
            mask_config: 1 if change the top half, 0 if change the bottom half.
            hps: the set of hyperparameters.
        """
        super(ChannelwiseAffineCoupling, self).__init__(mask_config, hps)

        self.scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.scale_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.in_bn = nn.BatchNorm2d(in_out_dim//2)
        self.block = nn.Sequential(        # 1st half of resnet: shift
            nn.ReLU(),                    # 2nd half of resnet: log_rescale
            ResidualModule(in_out_dim, mid_dim, in_out_dim, 
                self.res_blocks, self.bottleneck, self.skip, self.weight_norm))
        self.out_bn = nn.BatchNorm2d(in_out_dim//2, affine=False)

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log of diagonal elements of Jacobian.
        """
        [_, C, _, _] = list(x.size())
        if self.mask_config:
            (on, off) = x.split(C//2, dim=1)
        else:
            (off, on) = x.split(C//2, dim=1)
        off_ = self.in_bn(off)
        off_ = torch.cat((off_, -off_), dim=1)     # C channels
        out = self.block(off_)
        (shift, log_rescale) = out.split(C//2, dim=1)
        log_rescale = self.scale * torch.tanh(log_rescale) + self.scale_shift
        
        log_diag_J = log_rescale     # See Eq(6) in real NVP
        # See Eq(7) and Eq(8) and Section 3.7 in real NVP
        if reverse:
            if self.coupling_bn:
                mean, var = self.out_bn.running_mean, self.out_bn.running_var
                mean = mean.reshape(-1, 1, 1, 1).transpose(0, 1)
                var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                on = on * torch.exp(0.5 * torch.log(var + 1e-5)) + mean
            on = (on - shift) * torch.exp(-log_rescale)
        else:
            on = on * torch.exp(log_rescale) + shift
            if self.coupling_bn:
                if self.training:
                    _, var = self.batch_stat(on)
                else:
                    var = self.out_bn.running_var
                    var = var.reshape(-1, 1, 1, 1).transpose(0, 1)
                on = self.out_bn(on)
                log_diag_J = log_diag_J - 0.5 * torch.log(var + 1e-5)
        if self.mask_config:
            x = torch.cat((on, off), dim=1)
            log_diag_J = torch.cat((log_diag_J, torch.zeros_like(log_diag_J)), 
                dim=1)
        else:
            x = torch.cat((off, on), dim=1)
            log_diag_J = torch.cat((torch.zeros_like(log_diag_J), log_diag_J), 
                dim=1)
        return x, log_diag_J
