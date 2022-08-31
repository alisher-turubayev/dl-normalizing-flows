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
import torch.optim as optim
import torch.utils.data as torchdata
import torch.distributions as distributions

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from flow_realnvp import RealNVP
from modules_dcgan import Discriminator, Generator
from utils import (
    logit_transform,
    Hyperparameters,
    weights_init
)

import math
import os

# Training loop for RealNVP
def train_flow(
    epochs,
    num_workers,
    datapath,
    dataset_name,
    batch_size,
    image_size,
    channels,
    base_dim,
    res_blocks,
    output_dir,
    fresh,
    saved_path,
    lr, 
    weight_decay
    ):
    # Define image transforms
    #   Note that the tensors are not normalized; this is done during the training loop with `logit_transform` function
    data_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(),
        ]
    )

    # Use ImageFolder dataset class
    dataset = ImageFolder(
        root = datapath + '/' + dataset_name, 
        transform = data_transforms
    )

    # Because using the original dataset, even after prunning to ~47k images takes too much,
    #   we take a random split of 100 batches
    if len(dataset) > batch_size * 100:
        dataset, _ = torchdata.random_split(dataset, [batch_size * 100, len(dataset) - (batch_size * 100)])

    # Split is 90% training set / 10% validation set
    train_set_size = math.floor(len(dataset) * 0.9)
    train_set, valid_set = torchdata.random_split(dataset, [train_set_size, len(dataset) - train_set_size])

    # Set the training and validation set loaders
    train_loader = torchdata.DataLoader(
        train_set,
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = num_workers
    )
    valid_loader = torchdata.DataLoader(
        valid_set,
        batch_size = batch_size, 
        shuffle= True,
        num_workers = num_workers
    )

    # See if CUDA-enabled device is available; if not, train on CPU
    try:
        device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    except RuntimeError:
        device = torch.device('cpu')
    
    # Use normal distributions for the prior 
    prior = distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device), validate_args = False)

    # Initialize the model
    #   Note that the parameters such as whether to use bottleneck, skip architecture, weight normalization and batchnorm coupling layer output
    #   are set to True by default.
    #
    #   Additionally, additive coupling was removed entirely from the implementation, and only affine coupling is now available. Therefore, an
    #   associated flag was removed.
    model = RealNVP(
        channels, 
        image_size,
        prior,
        Hyperparameters(
            base_dim = base_dim,
            res_blocks = res_blocks,
            bottleneck = True,
            skip = True,
            weight_norm = True,
            coupling_bn = True
            )
        )
    
    model = model.to(device)
    # Use Adam optimizer
    #   Previously, an AdaMAX optimizer was used, but no discernable difference was found 
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # If the mode is not fresh (i.e. the training continues from the previous point), try loading the models
    #   Note that if either the model or optimizer failed to load, the program terminates - this was done because during training,
    #   using a fresh optimizer resulted in crashing/unexpected behaviour.
    if not fresh:
        if saved_path is None:
            print('Fresh mode was disabled, but the \'--saved-path\' was not specified. See -h/--help for help.')
            return
        try:
            model.load_state_dict(torch.load(os.path.join(saved_path, 'realnvp_state.pt')))
            print('Loaded saved model.')
        except Exception:
            print('Could not load \'realnvp_state.pt\' at {}, terminating.'.format(os.path.join(saved_path, 'realnvp_state.pt')))
            return
        try:
            optimizer.load_state_dict(torch.load(os.path.join(saved_path, 'realnvp_state_optim.pt')))
            print('Loaded saved optimizer.')
        except Exception:
            print('Could not load \'realnvp_state_optim.pt\' at {}, terminating.'.format(os.path.join(saved_path, 'realnvp_state_optim.pt')))
            return

    # oops - maybe need to move it to arguments?
    # Define scale regularization - this is the same default value as in @fmu2/realNVP
    scale_reg = 5e-5

    # Define training variables
    curr_epoch = 0
    optimal_logll = float('-inf')
    early_stop = 0

    # Start the training/validation loop
    while curr_epoch < epochs:
        curr_epoch += 1
        print('Current epoch: {}'.format(curr_epoch))

        # Before training loop, flush the variable
        running_logll = 0.

        # Training loop
        # Set the model into training mode
        model.train()
        for batch_idx, data in enumerate(train_loader):
            # Set the gradient of all optimized torch.Tensors to zero
            #   Further information is available here:
            #   https://stackoverflow.com/a/48009142
            optimizer.zero_grad()

            # Because we are working with unlabelled data, we can ignore the second unpacked variable
            x, _ = data

            # Perform a logit transform
            #   Note that log-determinant of Jacobian is returned for the input batch
            x, logdet = logit_transform(x)
            x = x.to(device)
            logdet = logdet.to(device)
            # Calculate log-likelihood 
            logll, weight_scale = model(x)
            logll = (logll + logdet).mean()
            # For RealNVP, there is L2 regularization on scaling factors
            loss = -logll + scale_reg * weight_scale
            # Add the current log-likelihood to the running log-likelihood  of the current training pass
            running_logll += logll.item()
            # Compute the gradient for the current log-likelihood 
            loss.backward()
            # Update the parameters of the model
            optimizer.step()

        # Calculate average loss and bits per dims over the training pass
        mean_logll = running_logll / (batch_idx + 1)
        mean_bits_per_dim = (-mean_logll + np.log(256.) * image_size * image_size * channels) / (image_size * image_size * channels * np.log(2.))           

        # Output the bits per dims to monitor training
        print('::Mean bits per dims: {}'.format(mean_bits_per_dim))

        # Before validation loop, flush the variables
        running_logll = 0. 
        
        # Validation loop
        # Set the model into validation mode
        model.eval()
        # Use no_grad to prevent gradient calculations during the validation loop
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                x, _ = data
                # Perform logit transform
                x, logdet = logit_transform(x)
                x = x.to(device)
                logdet = logdet.to(device)
                logll, weight_scale = model(x)
                logll = (logll + logdet).mean()
                # For RealNVP, there is L2 regularization on scaling factors
                loss = -logll + scale_reg * weight_scale
                running_logll += logll.item()
    
        # Calculate the average loss and bits per dims for the validation run
        mean_logll = running_logll / (batch_idx + 1)
        mean_bits_per_dim = (-mean_logll + np.log(256.) * image_size * image_size * channels) / (image_size * image_size * channels * np.log(2.))           

        print('::Mean validation bits per dims: {}'.format(mean_bits_per_dim))

        # Early stopping:
        #   Early stopping is implemented by checking whether the mean validation log-likelihood is larger than the optimal log-likelihood.
        #   However, due to the way training was conducted, this is not likely to be ever achieved.
        if mean_logll > optimal_logll:
            early_stop = 0
            optimal_logll = mean_logll
        else:
            early_stop += 1
            if early_stop >= 100:
                break
    
    print('Training finished at epoch {} with log-likelihood {}'.format(curr_epoch, optimal_logll))

    # Save the model and optimizer
    torch.save(model.state_dict(), os.path.join(output_dir, 'states', 'realnvp_state.pt'))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'states', 'realnvp_state_optim.pt'))

    # Generate samples
    with torch.no_grad():
        imgs, _ = logit_transform(
            model.sample(size = 100), 
            reverse = True
            )

        torchvision.utils.save_image(imgs, os.path.join(output_dir, 'gen', 'img_realnvp.png'), nrows = 10)
    return

def train_dcgan(
    epochs, 
    num_workers,
    datapath,
    dataset_name,
    batch_size,
    image_size,
    channels,  
    output_dir,
    nz, 
    ngf, 
    ndf, 
    lr,
    weight_decay,
    fresh,
    saved_path
    ):
    # Define image transforms
    #   Note that the tensors are normalized here to mean 0.5 and std 0.5 across all channels
    data_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)), 
            transforms.CenterCrop(image_size), 
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Use ImageFolder dataset class
    dataset = ImageFolder(
        root = datapath + '/' + dataset_name, 
        transform = data_transforms
    )

    # Because using the original dataset, even after prunning to ~47k images takes too much,
    # we take a random split of 100 batches
    if len(dataset) > batch_size * 100:
        dataset, _ = torchdata.random_split(dataset, [batch_size * 100, len(dataset) - (batch_size * 100)])

    # Set the training set loader - note the absence of the validation set.
    #   The original implementation by @inkawhich also did not have a validation set.
    train_loader = torchdata.DataLoader(
        dataset,
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = num_workers
    )

    # Check if CUDA-enabled device is available, and initialize appropriately
    if torch.cuda.is_available():
        ngpu = 1
        device = torch.device('cuda:0')
    else:
        ngpu = 0
        device = torch.device('cpu')

    # Define the generator and discriminator models 
    #   Additionally, both models are initialized with weights with mean 0 and std 0.2 as in the DCGAN paper
    generator = Generator(ngpu, channels, nz, ngf)
    generator.apply(weights_init)
    
    discriminator = Discriminator(ngpu, channels, ndf)
    discriminator.apply(weights_init)

    # Hotfix for CUDA issue: https://stackoverflow.com/a/59013131
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    # Define the loss function - in this case Binary Cross Entropy
    #   Additional information: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    criterion = nn.BCELoss()

    # Define real and fake labels
    real_label = 1.
    fake_label = 0.

    # Define optimizers (AdaMAX was never tested)
    optimizer_gen = optim.Adam(generator.parameters(), lr = lr, weight_decay = weight_decay)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr = lr, weight_decay = weight_decay)

    # Attempt to load the previously saved models if requested
    if not fresh:
        if saved_path is None:
            print('Fresh mode was disabled, but the \'--saved-path\' was not specified. See -h/--help for help.')
            return
        try:
            generator.load_state_dict(torch.load(os.path.join(saved_path, 'generator_state.pt')))
            discriminator.load_state_dict(torch.load(os.path.join(saved_path, 'discriminator_state.pt')))
            print('Loaded saved models.')
        except Exception:
            print('Could not load \'generator_state.pt\' and/or \'discriminator_state.pt\' at {}, terminating.'.format(saved_path))
            return
        try:
            optimizer_gen.load_state_dict(torch.load(os.path.join(saved_path, 'generator_state_optim.pt')))
            optimizer_disc.load_state_dict(torch.load(os.path.join(saved_path, 'discriminator_state_optim.pt')))
            print('Loaded saved optimizers.')
        except Exception:
            print('Could not load \'generator_state_optim.pt\' and/or \'discriminator_state_optim.pt\' at {}, terminating.'.format(saved_path))
            return

    curr_epoch = 0
    while curr_epoch < epochs:
        curr_epoch += 1
        print('Current epoch: {}'.format(curr_epoch))

        # Average loss for generator / discriminator
        mean_err_disc = 0.
        mean_err_gen = 0.

        for batch_idx, data in enumerate(train_loader):
            # Set gradients to all model parameters to 0
            discriminator.zero_grad()

            x, _ = data
            x = x.to(device)

            # This is a hotfix - sometimes (especially with small testing dataset used to test the program locally),
            #   the batch size would not be equal to 64.
            provided_batch_size = x.size(0)

            # Fill the label data with real label (because data is coming from training set)
            label = torch.full((provided_batch_size,), real_label, dtype = torch.float, device = device)

            # Get the output of the discriminator (sigmoid activation function - predicted labels)
            #   view(-1) flattens the tensor: https://discuss.pytorch.org/t/what-does-view-1-do/109803
            output = discriminator(x).view(-1)

            # Use the criterion to calculate loss on classifying real data 
            err_disc_real = criterion(output, label)
            err_disc_real.backward()

            # Generate random noise to feed the generator
            noise = torch.randn(provided_batch_size, nz, 1, 1, device = device)
            
            # Generate fake images
            fake = generator(noise)
            label.fill_(fake_label)

            # Get the output of the discriminator on fake data
            output = discriminator(fake.detach()).view(-1)

            # Calculate loss on classifying fake data
            err_disc_fake = criterion(output, label)
            err_disc_fake.backward()

            # Calculate the total error (error on real data and fake data) and add that to the running loss
            err_disc = err_disc_real + err_disc_fake
            mean_err_disc += err_disc.item()

            # Update the parameters of the discriminator model
            optimizer_disc.step()

            # Generator training step
            # Set gradients to all model parameters to 0
            generator.zero_grad()
            # Fill the labels back with real label (for generator, fake labels are real)
            label.fill_(real_label)

            # Discriminator was just updated above; do a pass of generated data through it
            output = discriminator(fake).view(-1)

            # Calculate loss of the generator based on the output of updated discriminator
            err_gen = criterion(output, label)
            err_gen.backward()

            mean_err_gen += err_gen.item()

            # Update the parameters of the generator model
            optimizer_gen.step()

        # Output statistics for monitoring
        print("::Mean loss for discriminator after epoch: {}".format(mean_err_disc / (batch_idx + 1)))
        print("::Mean loss for generator after epoch: {}".format(mean_err_gen / (batch_idx + 1)))
    
    # Save models 
    torch.save(generator.state_dict(), os.path.join(output_dir, 'states', 'generator_state.pt'))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, 'states', 'discriminator_state.pt'))
    torch.save(optimizer_gen.state_dict(), os.path.join(output_dir, 'states', 'generator_state_optim.pt'))
    torch.save(optimizer_disc.state_dict(), os.path.join(output_dir, 'states', 'discriminator_state_optim.pt'))

    # Define noise for image generation (similar to RealNVP, generate 100 images)
    fixed_noise = torch.randn(100, nz, 1, 1, device=device)

    # Save images
    with torch.no_grad():
        imgs = generator(fixed_noise).detach().cpu()
        torchvision.utils.save_image(
            torchvision.utils.make_grid(imgs, nrow = 10, normalize = True),
            os.path.join(output_dir, 'gen', 'img_dcgan.png')
        )
    return