# References:
#   https://github.com/ikostrikov/pytorch-flows/,
#   https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html,
#   https://github.com/fmu2/realNVP

import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as torchdata
import torch.distributions as distributions

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import argparse

from flow_realnvp import RealNVP
from utils import (
    logit_transform,
    Hyperparameters,
)

import os
import math

def main(
    algo,
    epochs,
    K,
    L,
    num_hidden,
    lr,
    weight_decay,
    dataset_name,
    datapath,
    batch_size,
    image_size,
    channels,
    num_workers,
    output_dir,
    fresh,
    saved_model,
    saved_optimizer,
    fixed
    ):
    # If the fixed mode is requested, set the seed to 999 for reproducibility
    if fixed:
        torch.manual_seed(999)

    # If the output directory is not specified, make it working directory
    if output_dir is None:
        output_dir = os.path.join(work_dir, 'outputs')

    # Create all required directories if needed
    try:
        os.makedirs(output_dir)
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(output_dir, 'states'))
    except OSError:
        pass

    try: 
        os.makedirs(os.path.join(output_dir, 'gen'))
    except OSError:
        pass

    try:
        os.makedirs(os.path.join(output_dir, 'checkpoints'))
    except OSError:
        pass

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
    elif algo == 'realnvp':
        # Use normal distributions for the prior 
        prior = distributions.Normal(torch.tensor(0.).to(device), torch.tensor(1.).to(device))

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
    else:
        print('Currently no implementation of GAN is available. :c')
        return
    
    optimizer = optim.Adamax(model.parameters(), lr = lr, weight_decay = weight_decay)
    
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

    model = model.to(device)

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
            else:
                pass
            
            optimizer.step()

        mean_logll = running_loss_nll / (batch_idx + 1)

        if algo == 'glow':
            pass
        elif algo == 'realnvp':
            mean_bits_per_dim = (-mean_logll + np.log(256.) * image_size * image_size * channels) / (image_size * image_size * channels * np.log(2.))           
        else:
            pass

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
                elif algo == 'realnvp':
                    x, logdet = logit_transform(x)
                    x = x.to(device)
                    logdet = logdet.to(device)
                    logll, weight_scale = model(x)
                    logll = (logll + logdet).mean()
                    # For RealNVP, there is L2 regularization on scaling factors
                    loss = -logll + scale_reg * weight_scale
                    running_loss_nll += logll.item()
                else:
                    pass
        
        mean_logll = running_loss_nll / (batch_idx + 1)

        if algo == 'glow':
            pass
        elif algo == 'realnvp':
            mean_bits_per_dim = (-mean_logll + np.log(256.) * image_size * image_size * channels) / (image_size * image_size * channels * np.log(2.))           
        else:
            pass

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
        elif algo == 'realnvp':
            imgs, _ = logit_transform(
                model.sample(size = 100), 
                reverse = True
                )
        else:
            pass

        torchvision.utils.save_image(imgs, output_dir + '/gen/img_' + algo + '.png', nrows = 10)
    return

if __name__ == "__main__":
    # Get the current directory path
    work_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description = 'Utilizing Normalizing Flows for Anime Face Generation - Main Program')

    parser.add_argument(
        '--algo',
        type = str,
        default = 'glow',
        choices = ['glow', 'realnvp', 'gan'],
        help = 'The type of algorithm to train. Default is Glow.'
    )

    parser.add_argument(
        '--epochs',
        type = int,
        default = 500,
        help = 'The number of epochs to train the model. Default is 500.'
    )

    parser.add_argument(
        '--blocks-per-level',
        type = int,
        dest = 'K',
        default = 32,
        help = 'The number of blocks per each level in a Glow model. By default is set to 32. Ignored when \'--algo\' argument is \'realnvp\' or \'gan\'.'
    )

    parser.add_argument(
        '--levels',
        type = int,
        dest = 'L',
        default = 3,
        help = 'The number of levels in a Glow model. Ignored when \'--algo\' argument is \'realnvp\' or \'gan\'.'
    )

    parser.add_argument(
        '--num-hidden',
        type = int,
        default = 512,
        help = 'Number of hidden channels in the ActNorm layer (used for convolutions within the layer as in the Glow paper). Default is 512. Ignored when \'--algo\' argument is \'real-nvp\' or \'gan\'.'
    )

    parser.add_argument(
        '--learning-rate',
        type = float,
        dest = 'lr',
        default = 5e-4,
        help = 'Learning rate for the model. By default 5e-4 or 0.0005.'
    )

    parser.add_argument(
        '--weight-decay',
        type = float,
        default = 5e-5,
        help = 'Weight decay for the model. By default 5e-5 or 0.00005.'
    )

    parser.add_argument(
        '--dataset-name', 
        type = str, 
        default = 'kaggle-full',  
        help = 'The dataset to be used. By default, \'Anime Faces Dataset\' from Kaggle is used. Custom datasets can be unzipped into \'datasets\' folder - see \'datasets/README.md\' for instructions.'
    )

    parser.add_argument(
        '--datapath',
        type = str,
        default = work_dir + '/datasets',
        help = 'The path to the dataset. By default, *current working directory*/\'datasets\' is used. Custom datasets can be unzipped into \'datasets\' folder - see \'datasets/README.md\' for instructions.'
    )

    parser.add_argument(
        '--batch-size',
        type = int,
        default = 64,
        help = 'The batch size used during training. By default 64.'
    )

    parser.add_argument(
        '--image-size',
        type = int,
        default = 64,
        help = 'Image size of each image. By default, set to 64 (meaning that images are square and 64x64 in dimensions).'
    )

    parser.add_argument(
        '--channels',
        type = int,
        default = 3,
        help = 'Number of channels in each image. By default, 3 (for RGB).'
    )

    parser.add_argument(
        '--num-workers',
        type = int,
        default = 2,
        help = 'Number of workers in the Dataloader. Default is 2.'
    )

    parser.add_argument(
        '--output-dir',
        type = str,
        help = 'Output directory for all generated data (model states, checkpoints, generated data). Default is *current working directory*/\'outputs\''
    )

    parser.add_argument(
        '--nofresh',
        action = 'store_true',
        help = 'Should the model be trained from the scratch - if you specify this argument, the model will trained further. Only change this if you have trained a model before.'
    )

    parser.add_argument(
        '--saved-model',
        type = str,
        help = 'The file that contains the previously trained model. Include the path, not just the name of the model file.'
    )

    parser.add_argument(
        '--saved-optimizer',
        type = str,
        help = 'The file that contains the previously trained optimizer. Include the path, not just the name of the model optimizer.'
    )

    parser.add_argument(
        '--nofixed',
        action = 'store_true',
        help = 'Should the seed be fixed for reproducibility - if you specify this argument, the model will be trained with a random seed. Defaults to true and is the seed is set to 999.'
    )

    args = parser.parse_args()
    kwargs = vars(args)

    if (args.nofresh):
        kwargs['fresh'] = False
    else:
        kwargs['fresh'] = True

    if (args.nofixed):
        kwargs['fixed'] = False
    else:
        kwargs['fixed'] = True

    del kwargs['nofresh']
    del kwargs['nofixed']

    main(**kwargs)
