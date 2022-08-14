# Stolen from https://github.com/y0ast/Glow-PyTorch/blob/master/model.py for now
# TODO: rewrite from scratch - understanding each nook and cranny

import math

import torch
import torch.nn as nn

from modules import (
    SqueezeLayer,
    Split2d,
    ActNorm,
    InvConv1x1,
    affine_coupling,
    Conv2dZeros,
)

from utils import (
    split_cross_feature, 
    uniform_binning_correction, 
    gaussian_likelihood, 
    gaussian_sample
)

class FlowStep(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels
    ):    
        super().__init__()
        # For definition of the flow step, see figure 2(a) of https://arxiv.org/pdf/1807.03039.pdf
        
        # First, Activisation Normalization
        self.actnorm = ActNorm(in_channels)
        # Second, Invertible 1x1 Convolution with LU decomposition
        self.invconv = InvConv1x1(in_channels)
        self.flow_permutation = lambda z, logdet, reverse = False: self.invconv(z, logdet, reverse)
        # Last, Affine Coupling
        self.coupling = affine_coupling(in_channels // 2, in_channels, hidden_channels)

    def forward(self, input, logdet = None, reverse = False):
        if not reverse:
            return self.normal_flow(input, logdet)
        if reverse:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0
        # ActNorm
        z, logdet = self.actnorm(input, logdet)
        # InvConv
        z, logdet = self.flow_permutation(z, logdet)
        # Coupling
        #   Split along the channels
        z1, z2 = split_cross_feature(z, 'split')
        #   Feed thru the NN (shallow neural network)
        h = self.coupling(z1)
        #   Calculate shift and scale for the first split *half*
        shift, scale = split_cross_feature(h, 'cross')
        scale = torch.sigmoid(scale + 2.0)
        #   Apply shift and scale
        z2 = z2 + shift
        z2 = z2 * scale
        #   Calculate log-determinants 
        logdet = torch.sum(torch.log(scale), dim = [1, 2, 3]) + logdet
        #   Apply concatenation operation
        z = torch.cat((z1, z2), dim = 1)
        # Return the output
        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # Affine Coupling
        #   Split along the channels
        z1, z2 = split_cross_feature(input, 'split')
        #   Feed thru the NN (shallow neural network)
        h = self.coupling(z1)
        #   Calculate shift and scale for the first split *half*
        shift, scale = split_cross_feature(h, 'cross')
        scale = torch.sigmoid(scale + 2.0)
        #   Reverse the operation (apply shift and scale in forward flow)
        z2 = z2 / scale
        z2 = z2 - shift
        #   Subtract the previously added log-determinant 
        logdet = -torch.sum(torch.log(scale), dim = [1, 2, 3]) + logdet
        #   Apply concatenation operation
        z = torch.cat((z1, z2), dim = 1)
        # InvConv
        z, logdet = self.flow_permutation(z, logdet, reverse = True)
        # ActNorm
        z, logdet = self.actnorm(z, logdet, reverse = True)
        # Return the reversed input
        return z, logdet

class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(FlowStep(C, hidden_channels))
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2
    
    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, _ in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, Split2d):
                z, _ = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, _ = layer(z, logdet=0, reverse=True)
        return z

class Glow(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        K,
        L,
        learn_top = True,
    ):
        super().__init__()
        self.flow = FlowNet(
            image_shape=image_shape,
            hidden_channels=hidden_channels,
            K=K,
            L=L
        )
        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = Conv2dZeros(C * 2, C * 2)

        self.register_buffer(
            "prior_h",
            torch.zeros(
                [
                    1,
                    self.flow.output_shapes[-1][1] * 2,
                    self.flow.output_shapes[-1][2],
                    self.flow.output_shapes[-1][3],
                ]
            ),
        )

    def prior(self, data):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            h = self.prior_h.repeat(100, 1, 1, 1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        return split_cross_feature(h, "split")

    def forward(self, x=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, temperature)
        else:
            return self.normal_flow(x)

    def normal_flow(self, x):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x)
        objective += gaussian_likelihood(mean, logs, z)
       
        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd

    def reverse_flow(self, z, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm):
                m.inited = True
    
    def sample(self, temperature):
        return self.forward(temperature = temperature, reverse = True)