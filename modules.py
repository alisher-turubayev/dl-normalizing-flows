# Stolen from https://github.com/y0ast/Glow-PyTorch/blob/master/modules.py for now
# TODO: rewrite

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    calculate_same_padding, 
    split_cross_feature, 
    unsqueeze2d, 
    squeeze2d,
    gaussian_sample,
    gaussian_likelihood    
)

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False):
        if reverse:
            output = unsqueeze2d(input, self.factor)
        else:
            output = squeeze2d(input, self.factor)

        return output, logdet

class Split2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        return split_cross_feature(h, "cross")

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = split_cross_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet

class ActNorm(nn.Module):
    def __init__(
        self, 
        num_features,
    ):
        super().__init__()
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError('In evaluation mode, but ActNorm is not initialized')
        
        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim = [0, 2, 3], keepdim = True)
            vars = torch.mean((input.clone() + bias) ** 2, dim = [0, 2, 3], keepdim = True)
            logs = torch.log(1.0 / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            
            self.inited = True

    def forward(self, input, logdet = None, reverse = False):
        if not self.inited:
            self.initialize_parameters(input)

        if not reverse:
            input = input + self.bias
            input = input * torch.exp(self.logs)

            if logdet is not None:
                _, _, h, w = input.shape
                dlogdet = torch.sum(self.logs) * h * w
                logdet = logdet + dlogdet
        else:
            input = input * torch.exp(-self.logs)
            if logdet is not None:
                _, _, h, w = input.shape
                dlogdet = -1 * torch.sum(self.logs) * h * w
                logdet = logdet + dlogdet
            
            input = input - self.bias
        
        return input, logdet

class InvConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = torch.linalg.qr(torch.randn(*w_shape), mode = 'reduced')[0]
        p, lower, upper = torch.lu_unpack(*torch.lu(w_init))
        s = torch.diag(upper)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))
        upper = torch.triu(upper, 1)
        l_mask = torch.tril(torch.ones(w_shape), -1)
        eye = torch.eye(*w_shape)
        self.register_buffer("p", p)
        self.register_buffer("sign_s", sign_s)
        self.lower = nn.Parameter(lower)
        self.log_s = nn.Parameter(log_s)
        self.upper = nn.Parameter(upper)
        self.l_mask = l_mask
        self.eye = eye
        self.w_shape = w_shape
 
    def get_weight(self, input, reverse):
        _, _, h, w = input.shape

        self.l_mask = self.l_mask.to(self.lower.device)
        self.eye = self.eye.to(self.lower.device)

        lower = self.lower * self.l_mask + self.eye

        u = self.upper * self.l_mask.transpose(0, 1).contiguous().to(self.upper.device)
        u += torch.diag(self.sign_s * torch.exp(self.log_s))

        dlogdet = torch.sum(self.log_s) * h * w

        if reverse:
            u_inv = torch.inverse(u)
            l_inv = torch.inverse(lower)
            p_inv = torch.inverse(self.p)

            weight = torch.matmul(u_inv, torch.matmul(l_inv, p_inv))
        else:
            weight = torch.matmul(self.p, torch.matmul(lower, u))

        return weight.view(self.w_shape[0], self.w_shape[1], 1, 1).to(input.device), dlogdet.to(input.device)

    def forward(self, input, logdet=None, reverse=False):
        weight, dlogdet = self.get_weight(input, reverse)

        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

def affine_coupling(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace = False),
        Conv2d(hidden_channels, hidden_channels, kernel_size = (1, 1)),
        nn.ReLU(inplace = False),
        Conv2dZeros(hidden_channels, out_channels)
    )
    return block

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        do_actnorm=True,
        weight_std=0.05,
    ):
        super().__init__()

        if padding == "same":
            padding = calculate_same_padding(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=(not do_actnorm),
        )

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.conv.bias.data.zero_()
        else:
            self.actnorm = ActNorm(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = self.conv(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x

class Conv2dZeros(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        logscale_factor=3,
    ):
        super().__init__()

        if padding == "same":
            padding = calculate_same_padding(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)
