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
import torch.nn as nn

# Generator Code
class Generator(nn.Module):
    # nz is the size of the latent space vector
    # ngf is the size of feature maps in the generator
    def __init__(self, ngpu, channels, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is latent space vector z
            # First convolution layer ([nz] -> [ngf * 8, 4, 4] )
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # Second convolution layer ([ngf * 8, 4, 4] -> [ngf * 4, 8, 8])
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # Third convolution layer ([ngf * 8, 4, 4] -> [ngf * 2, 16, 16])
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # Fourth convolution layer ([ngf * 2, 16, 16] -> [ngf, 32, 32])
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # Fifth convolution layer ([ngf, 32, 32] -> [3, 64, 64])
            nn.ConvTranspose2d( ngf, channels, 4, 2, 1, bias=False),
            # Use tanh function to return the data to the range of [-1, 1]
            nn.Tanh()
        )
    # Defines the forward pass of the model
    #   As the main contains the sequential list of layers, we can simply call main's forward function
    def forward(self, input):
        return self.main(input)

# Discriminator Code
class Discriminator(nn.Module):
    def __init__(self, ngpu, channels, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Input is image x
            # First convolution layer ([3, 64, 64] -> [ndf, 32, 32])
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Second convolution layer ([ndf, 32, 32] -> [(ndf*2), 16, 16])
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Third convolution layer ([(ndf*2), 16, 16] -> [(ndf*4), 8, 8])
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Fourth convolution layer ([(ndf*4), 8, 8] -> [(ndf*8), 4, 4])
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Fifth convolution ([ndf * 8, 4, 4] -> [1])
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # Use Sigmoid activation function to output 1 or 0 (real vs fake data)
            nn.Sigmoid()
        )

    # Defines the forward pass of the model
    #   As the main contains the sequential list of layers, we can simply call main's forward function
    def forward(self, input):
        return self.main(input)