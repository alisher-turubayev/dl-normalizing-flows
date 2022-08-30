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

def train_flow(
    epochs,
    num_workers,
    datapath,
    dataset_name,
    batch_size,
    image_size,
    channels,
    K,
    L,
    num_hidden,
    base_dim,
    res_blocks,
    output_dir,
    fresh,
    saved_path,
    lr, 
    weight_decay
    ):
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
    # we take a random split of 100 batches
    if len(dataset) > batch_size * 100:
        dataset, _ = torchdata.random_split(dataset, [batch_size * 100, len(dataset) - (batch_size * 100)])

    # Split is 90% training set / 10% validation set
    train_set_size = math.floor(len(dataset) * 0.9)
    train_set, valid_set = torchdata.random_split(dataset, [train_set_size, len(dataset) - train_set_size])

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

    try:
        device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    except RuntimeError:
        device = torch.device('cpu')

    
    # Use normal distributions for the prior 
    prior = distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device), validate_args = False)

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
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    if not fresh:
        if saved_path is None:
            print('Fresh mode was disabled, but the path to saved model .pt file was not specified. See -h/--help for help.')
            return
        try:
            model.load_state_dict(torch.load(os.path.join(saved_path, 'realnvp_state.pt')))
            print('Loaded saved model.')
        except Exception:
            print('Could not load model at {}, terminating.'.format(os.path.join(saved_path, 'realnvp_state.pt')))
            return
        try:
            optimizer.load_state_dict(torch.load(os.path.join(saved_path, 'realnvp_state_optim.pt')))
            print('Loaded saved optimizer.')
        except Exception:
            print('Could not load optimizer at {}, using a new one.'.format(os.path.join(saved_path, 'realnvp_state_optim.pt')))
            return

    scale_reg = 5e-5

    # Define training variables
    curr_epoch = 0
    optimal_logll = float('-inf')
    early_stop = 0

    while curr_epoch < epochs:
        curr_epoch += 1
        print('Current epoch: {}'.format(curr_epoch))

        # Before training loop, flush the variables
        running_loss_nll = 0.

        # Training loop
        model.train()
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            x, _ = data

            # log-determinant of Jacobian from the logit transform
            x, logdet = logit_transform(x)
            x = x.to(device)
            logdet = logdet.to(device)
            logll, weight_scale = model(x)
            logll = (logll + logdet).mean()
            # For RealNVP, there is L2 regularization on scaling factors
            loss = -logll + scale_reg * weight_scale
            running_loss_nll += logll.item()
            loss.backward()
        
            optimizer.step()

        mean_logll = running_loss_nll / (batch_idx + 1)

        mean_bits_per_dim = (-mean_logll + np.log(256.) * image_size * image_size * channels) / (image_size * image_size * channels * np.log(2.))           

        print('::Mean bits per dims: {}'.format(mean_bits_per_dim))

        # Before validation loop, flush the variables
        running_loss_nll = 0. 
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                x, _ = data
                
                x, logdet = logit_transform(x)
                x = x.to(device)
                logdet = logdet.to(device)
                logll, weight_scale = model(x)
                logll = (logll + logdet).mean()
                # For RealNVP, there is L2 regularization on scaling factors
                loss = -logll + scale_reg * weight_scale
                running_loss_nll += logll.item()
    
        mean_logll = running_loss_nll / (batch_idx + 1)

        mean_bits_per_dim = (-mean_logll + np.log(256.) * image_size * image_size * channels) / (image_size * image_size * channels * np.log(2.))           

        print('::Mean validation bits per dims: {}'.format(mean_bits_per_dim))

        if mean_logll > optimal_logll:
            early_stop = 0
            optimal_logll = mean_logll
        else:
            early_stop += 1
            if early_stop >= 100:
                break
    
    print('Training finished at epoch {} with log-likelihood {}'.format(curr_epoch, optimal_logll))

    torch.save(model.state_dict(), os.path.join(output_dir, 'states', 'realnvp_state.pt'))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'states', 'realnvp_state_optim.pt'))

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

    train_loader = torchdata.DataLoader(
        dataset,
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = num_workers
    )

    if torch.cuda.is_available():
        ngpu = 1
        device = torch.device('cuda:0')
    else:
        ngpu = 0
        device = torch.device('cpu')

    generator = Generator(ngpu, channels, nz, ngf)
    generator.apply(weights_init)
    
    discriminator = Discriminator(ngpu, channels, ndf)
    discriminator.apply(weights_init)

    # Hotfix for CUDA issue: https://stackoverflow.com/a/59013131
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizer_gen = optim.Adam(generator.parameters(), lr = lr, weight_decay = weight_decay)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr = lr, weight_decay = weight_decay)

    if not fresh:
        if saved_path is None:
            print('Fresh mode was disabled, but the model .pt file was not specified. See -h/--help for help.')
            return
        try:
            generator.load_state_dict(torch.load(os.path.join(saved_path, 'generator_state.pt')))
            discriminator.load_state_dict(torch.load(os.path.join(saved_path, 'discriminator_state.pt')))
            print('Loaded saved model.')
        except Exception:
            print('Could not load model at {}, terminating.'.format(saved_path))
            return
        try:
            optimizer_gen.load_state_dict(torch.load(os.path.join(saved_path, 'generator_state_optim.pt')))
            optimizer_disc.load_state_dict(torch.load(os.path.join(saved_path, 'discriminator_state_optim.pt')))
            print('Loaded saved optimizer.')
        except Exception:
            print('Could not load optimizer at {}, terminating.'.format(saved_path))
            return

    curr_epoch = 0
    while curr_epoch < epochs:
        curr_epoch += 1
        print('Current epoch: {}'.format(curr_epoch))

        # Average loss for generator / discriminator
        mean_err_disc = 0.
        mean_err_gen = 0.

        for batch_idx, data in enumerate(train_loader):
            discriminator.zero_grad()

            real_cpu = data[0].to(device)
            provided_batch_size = real_cpu.size(0)

            label = torch.full((provided_batch_size,), real_label, dtype = torch.float, device = device)

            output = discriminator(real_cpu).view(-1)

            err_disc_real = criterion(output, label)
            err_disc_real.backward()

            noise = torch.randn(provided_batch_size, nz, 1, 1, device = device)
            
            fake = generator(noise)
            label.fill_(fake_label)

            output = discriminator(fake.detach()).view(-1)

            err_disc_fake = criterion(output, label)
            err_disc_fake.backward()

            err_disc = err_disc_real + err_disc_fake
            mean_err_disc += err_disc.item()

            optimizer_disc.step()

            # Generator training step
            generator.zero_grad()
            label.fill_(real_label)

            output = discriminator(fake).view(-1)

            err_gen = criterion(output, label)
            err_gen.backward()

            mean_err_gen += err_gen.item()

            optimizer_gen.step()

        print("::Mean loss for discriminator after epoch: {}".format(mean_err_disc / (batch_idx + 1)))
        print("::Mean loss for generator after epoch: {}".format(mean_err_gen / (batch_idx + 1)))
    
    # Save models 
    torch.save(generator.state_dict(), os.path.join(output_dir, 'states', 'generator_state.pt'))
    torch.save(discriminator.state_dict(), os.path.join(output_dir, 'states', 'discriminator_state.pt'))
    torch.save(optimizer_gen.state_dict(), os.path.join(output_dir, 'states', 'generator_state_optim.pt'))
    torch.save(optimizer_disc.state_dict(), os.path.join(output_dir, 'states', 'discriminator_state_optim.pt'))

    # Save images on fixed noise
    fixed_noise = torch.randn(100, nz, 1, 1, device=device)

    with torch.no_grad():
        imgs = generator(fixed_noise).detach().cpu()
        torchvision.utils.save_image(imgs, os.path.join(output_dir, 'gen', 'img_dcgan.png'), nrows = 10)
    return