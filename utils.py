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
import torch.distributions as distributions

# RealNVP utility class/function - https://github.com/fmu2/realNVP
def logit_transform(x, constraint=0.9, reverse=False):
    '''Transforms data from [0, 1] into unbounded space.
    Restricts data into [0.05, 0.95].
    Calculates logit(alpha+(1-alpha)*x).
    Args:
        x: input tensor.
        constraint: data constraint before logit.
        reverse: True if transform data back to [0, 1].
    Returns:
        transformed tensor and log-determinant of Jacobian from the transform.
        (if reverse=True, no log-determinant is returned.)
    '''
    if reverse:
        x = 1. / (torch.exp(-x) + 1.)    # [0.05, 0.95]
        x *= 2.             # [0.1, 1.9]
        x -= 1.             # [-0.9, 0.9]
        x /= constraint     # [-1, 1]
        x += 1.             # [0, 2]
        x /= 2.             # [0, 1]
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        x = (x * 255. + noise) / 256.
        
        # restrict data
        x *= 2.             # [0, 2]
        x -= 1.             # [-1, 1]
        x *= constraint     # [-0.9, 0.9]
        x += 1.             # [0.1, 1.9]
        x /= 2.             # [0.05, 0.95]

        # logit data
        logit_x = torch.log(x) - torch.log(1. - x)

        # log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))

class Hyperparameters():
    def __init__(
        self, 
        base_dim, 
        res_blocks, 
        bottleneck, 
        skip, 
        weight_norm, 
        coupling_bn
    ):
        """Instantiates a set of hyperparameters used for constructing layers.
        Args:
            base_dim: features in residual blocks of first few layers.
            res_blocks: number of residual blocks to use.
            bottleneck: True if use bottleneck, False otherwise.
            skip: True if use skip architecture, False otherwise.
            weight_norm: True if apply weight normalization, False otherwise.
            coupling_bn: True if batchnorm coupling layer output, False otherwise.
        """
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn

# DCGAN utility function - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if torch.cuda.is_available():
            nn.init.normal_(m.weight.data, 0.0, 0.02).cuda()
        else:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        if torch.cuda.is_available():
            nn.init.normal_(m.weight.data, 1.0, 0.02).cuda()
            nn.init.constant_(m.bias.data, 0).cuda()
        else:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)