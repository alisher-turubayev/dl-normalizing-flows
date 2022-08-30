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

def train_flow(
    algo,
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
    output_dir,
    fresh,
    saved_model,
    saved_optimizer,
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

    if algo == 'glow':
        print('Currently no implementation of Glow is available. :c')
        return
    else:
        # Use normal distributions for the prior 
        prior = distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device), validate_args = False)

        model = RealNVP(
            channels, 
            image_size,
            prior,
            Hyperparameters(
                base_dim = 32,
                res_blocks = 8,
                bottleneck = True,
                skip = True,
                weight_norm = True,
                coupling_bn = True
                )
            )
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    if not fresh:
        if saved_model is None:
            print('Fresh mode was disabled, but the model .pt file was not specified. See -h/--help for help.')
            return
        try:
            model.load_state_dict(torch.load(saved_model))
            print('Loaded saved model.')
        except Exception:
            print('Could not load model at {}, terminating.'.format(saved_model))
            return
        if optimizer is None:
            print('Stored optimizer not specified, using a new one.')
        else:
            try:
                optimizer.load_state_dict(torch.load(saved_optimizer))
                print('Loaded saved optimizer.')
            except Exception:
                print('Could not load optimizer at {}, using a new one.'.format(saved_optimizer))
                pass

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

            if algo == 'glow':
                pass
            elif algo == 'realnvp':
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

        if algo == 'glow':
            pass
        else:
            mean_bits_per_dim = (-mean_logll + np.log(256.) * image_size * image_size * channels) / (image_size * image_size * channels * np.log(2.))           

        print('::Mean bits per dims: {}'.format(mean_bits_per_dim))

        # Before validation loop, flush the variables
        running_loss_nll = 0. 
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                x, _ = data
                
                if algo == 'glow':
                    pass
                else:
                    x, logdet = logit_transform(x)
                    x = x.to(device)
                    logdet = logdet.to(device)
                    logll, weight_scale = model(x)
                    logll = (logll + logdet).mean()
                    # For RealNVP, there is L2 regularization on scaling factors
                    loss = -logll + scale_reg * weight_scale
                    running_loss_nll += logll.item()
        
        mean_logll = running_loss_nll / (batch_idx + 1)

        if algo == 'glow':
            pass
        else:
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

    torch.save(model.state_dict(), output_dir + '/states/' + algo + '_state.pt')
    torch.save(optimizer.state_dict(), output_dir + '/states/' + algo + '_state_optim.pt')

    with torch.no_grad():
        imgs = []

        if algo == 'glow':
            pass
        else:
            imgs, _ = logit_transform(
                model.sample(size = 100), 
                reverse = True
                )

        torchvision.utils.save_image(imgs, output_dir + '/gen/img_' + algo + '.png', nrows = 10)
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
    discriminator = Discriminator(ngpu, channels, ndf)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    optimizer_gen = optim.Adam(generator.parameters(), lr = lr, weight_decay = weight_decay)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr = lr, weight_decay = weight_decay)

    curr_epoch = 0
    while curr_epoch < epochs:
        curr_epoch += 1
        print('Current epoch: {}'.format(curr_epoch))

        for _, data in enumerate(train_loader):
            discriminator.zero_grad()

            real_cpu = data[0].to(device)
            provided_batch_size = real_cpu.size(0)

            label = torch.full((provided_batch_size,), real_label, dtype=torch.float, device=device)

            output = discriminator(real_cpu).view(-1)

            err_disc_real = criterion(output, label)
            err_disc_real.backward()

            noise = torch.randn(provided_batch_size, nz, 1, 1, device=device)
            
            fake = generator(noise)
            label.fill_(fake_label)

            output = discriminator(fake.detach()).view(-1)

            err_disc_fake = criterion(output, label)
            err_disc_fake.backward()

            err_disc = err_disc_real + err_disc_fake
            optimizer_disc.step()

            # Generator training step
            generator.zero_grad()
            label.fill_(real_label)

            output = discriminator(fake).view(-1)

            err_gen = criterion(output, label)
            err_gen.backward()

            optimizer_gen.step()

        print("::Loss for discriminator after epoch: {}".format(err_disc.item()))
        print("::Loss for generator after epoch: {}".format(err_gen.item()))
    
    # Save images on fixed noise
    fixed_noise = torch.randn(100, nz, 1, 1, device=device)

    with torch.no_grad():
        imgs = generator(fixed_noise).detach().cpu()
        torchvision.utils.save_image(imgs, output_dir + '/gen/img_dcgan.png', nrows = 10)
    return