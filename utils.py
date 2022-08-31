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
# Function to perform a logit transform of the input data
#   This was described in the RealNVP paper in section 4.1
def logit_transform(x, constraint=0.9, reverse=False):
    if reverse:
        # Apply the reverse of the operations
        x = 1. / (torch.exp(-x) + 1.)    
        x *= 2. 
        x -= 1. 
        x /= constraint 
        x += 1. 
        x /= 2. 
        return x, 0
    else:
        [B, C, H, W] = list(x.size())
        
        # Generate noise for dequantization
        noise = distributions.Uniform(0., 1.).sample((B, C, H, W))
        # Normalize data
        x = (x * 255. + noise) / 256.
        
        # Restrict data:
        # [0, 1] -> [0, 2]
        # [0, 2] -> [-1, 1]
        # [-1, 1] -> [-0.9, 0.9]
        # [-0.9, 0.9] -> [0.1, 1.9]
        # [0.1, 1.9] -> [0.05, 0.95]
        x *= 2. 
        x -= 1. 
        x *= constraint 
        x += 1. 
        x /= 2. 

        # Apply logit operation on the data
        logit_x = torch.log(x) - torch.log(1. - x)

        # Calculate the log-determinant of Jacobian from the transform
        pre_logit_scale = torch.tensor(
            np.log(constraint) - np.log(1. - constraint))
        log_diag_J = F.softplus(logit_x) + F.softplus(-logit_x) \
            - F.softplus(-pre_logit_scale)

        return logit_x, torch.sum(log_diag_J, dim=(1, 2, 3))

# Class defining hyperparameters for the RealNVP
#   While only base_dim and res_blocks are changeable by the user, the rest of the arguments are left as-is regardless;
#   this is due to a large number of subclasses utilizing this support class.
#   However, the argument for affine/additive coupling was removed as it was easier to do so.
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
        self.base_dim = base_dim
        self.res_blocks = res_blocks
        self.bottleneck = bottleneck
        self.skip = skip
        self.weight_norm = weight_norm
        self.coupling_bn = coupling_bn

# DCGAN utility function - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Function to initialize weights with mean 0 and std 0.2
#   This is described in the section 4 of the DCGAN paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # This explicit check was added as a fix to local training (the author did not posses a discrete GPU)
        if torch.cuda.is_available():
            nn.init.normal_(m.weight.data, 0.0, 0.02).cuda()
        else:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # This explicit check was added as a fix to local training (the author did not posses a discrete GPU)
        if torch.cuda.is_available():
            nn.init.normal_(m.weight.data, 1.0, 0.02).cuda()
            nn.init.constant_(m.bias.data, 0).cuda()
        else:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)