# References:
#   https://github.com/ikostrikov/pytorch-flows/,
#   https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html,
#   https://github.com/fmu2/realNVP

import argparse
import torch

from train import train_dcgan, train_flow

import os

def main(
    algo,
    epochs,
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
    saved_path,
    fixed,
    base_dim,
    res_blocks,
    nz,
    ngf,
    ndf
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

    if algo == 'gan':
        train_dcgan(
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
        )
    else:
        train_flow(
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
        )


if __name__ == "__main__":
    # Get the current directory path
    work_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description = 'Utilizing Normalizing Flows for Anime Face Generation - Main Program')

    parser.add_argument(
        '--algo',
        type = str,
        default = 'realnvp',
        choices = ['realnvp', 'gan'],
        help = 'The type of algorithm to train. Default is \'realnvp\'.'
    )

    parser.add_argument(
        '--epochs',
        type = int,
        default = 500,
        help = 'The number of epochs to train the model. Default is 500.'
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
        '--saved-path',
        type = str,
        help = 'The file that contains the previously trained model/optimizer. Include the path without the file name/extension.'
    )

    parser.add_argument(
        '--nofixed',
        action = 'store_true',
        help = 'Should the seed be fixed for reproducibility - if you specify this argument, the model will be trained with a random seed. Defaults to true and is the seed is set to 999.'
    )

    # Arguments for RealNVP

    parser.add_argument(
        '--base-dim',
        type = int,
        default = 64,
        help = 'Features in residual blocks. Default is 64.'
    )

    parser.add_argument(
        '--res-blocks',
        type = int,
        default = 8,
        help = 'Number of residual blocks. Default is 8.'
    )

    # Arguments for DRAGAN
    parser.add_argument(
        '-nz', '--size-latent',
        type = int,
        dest = 'nz',
        default = 100,
        help = 'Size of z latent vector (i.e. size of generator input). Default is 100.'
    )

    parser.add_argument(
        '-ngf', '--size-feature-gen',
        type = int,
        dest = 'ngf',
        default = 64, 
        help = 'Size of feature maps in generator. Default is 64.'
    )

    parser.add_argument(
        '-ndf', '--size-feature-disc',
        type = int,
        dest = 'ndf',
        default = 64,
        help = 'Size of feature maps in discriminator. Default is 64.'
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
