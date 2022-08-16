# File is a mish-mash from https://github.com/ikostrikov/pytorch-flows/,
#   https://github.com/y0ast/Glow-PyTorch/ and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# 
# TODO: rewrite 

import torch
import torch.optim as optim
import torch.utils.data as torchdata

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

import argparse

import flows as fnn
from utils import compute_loss

import os

def main(
    algo,
    epochs,
    K,
    L,
    num_hidden,
    lr,
    weight_decay,
    warmup,
    dataset_name,
    datapath,
    batch_size,
    image_size,
    channels,
    num_workers,
    output_dir,
    fresh,
    fixed
    ):
    # If the fixed mode is requested, set the seed to 999 for reproducibility
    if fixed:
        torch.manual_seed(999)

    # Use ImageFolder dataset class
    dataset = ImageFolder(
        root = datapath + '/' + dataset_name, 
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)), 
                transforms.CenterCrop(image_size), 
                transforms.ToTensor(),
            ]
        )
    )

    # Because using the original dataset, even after prunning to ~47k images takes too much,
    # we take a random split of 100 batches

    subset = None
    if len(dataset) > batch_size * 100:
        subset, _ = torchdata.random_split(dataset, [batch_size * 100, len(dataset) - (batch_size * 100)])

    dataloader = torchdata.DataLoader(
        subset if (subset is not None) else dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = num_workers
    )

    # Hotfix - if the Runtime error raises, CUDA alloc on Google Colab wasn't possible
    # TODO: need to check if a better solution is available
    try:
        device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    except RuntimeError:
        device = torch.device('cpu')

    if algo == 'glow':
        model = fnn.Glow(
            (image_size, image_size, channels),
            num_hidden,
            K = K, 
            L = L,
            )
    elif algo == 'realnvp':
        print('Currently no implementation of RealNVP is available. :c')
        return
    else:
        print('Currently no implementation of StyleGAN is available. :c')
        return
    
    model = model.to(device)

    optimizer = optim.Adamax(model.parameters(), lr = lr, weight_decay = weight_decay)
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa

    train(epochs, device, model, dataloader, optimizer, lr_lambda, output_dir, fresh)

    try:
        os.makedirs('states')
    except OSError:
        pass

    torch.save(model.state_dict(), work_dir + '/states/' + algo + '_state.pt')
    torch.save(optimizer.state_dict(), work_dir + '/states/' + algo + '_state_optim.pt')

    with torch.no_grad():
        imgs = model.sample(temperature = 0.1).detach().cpu()

        try:
            os.makedirs('output')
        except OSError:
            pass

        torchvision.utils.save_image(imgs, 'output/img_' + algo + '.png', nrows = 10)
    return

def train(
    epochs, 
    device, 
    model, 
    dataloader, 
    optimizer, 
    lr_lambda, 
    output_dir, 
    fresh = True,
    saved_model = None, 
    saved_optimizer = None
    ):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, _ = batch
        x = x.to(device)

        _, nll = model(x)
        losses = compute_loss(nll)

        losses["total_loss"].backward()

        optimizer.step()

        return losses

    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "glow", n_saved=2, require_empty=False
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        {"model": model, "optimizer": optimizer},
    )

    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    if not fresh:
        model.load_state_dict(torch.load(saved_model))
        model.set_actnorm_init()

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(saved_model)
        resume_epoch = int(file_name.split("_")[-1])

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(
            f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
        )
        timer.reset()

    trainer.run(dataloader, epochs)

if __name__ == "__main__":
    # Get the current directory path
    work_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description = 'Utilizing Normalizing Flows for Anime Face Generation - Main Program')

    parser.add_argument(
        '--algo',
        type = str,
        default = 'glow',
        choices = ['glow', 'realnvp', 'style-gan'],
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
        help = 'The number of blocks per each level in a Glow model. By default is set to 32. Ignored when \'--algo\' argument is \'realnvp\' or \'style-gan\'.'
    )

    parser.add_argument(
        '--levels',
        type = int,
        dest = 'L',
        default = 3,
        help = 'The number of levels in a Glow model. Ignored when \'--algo\' argument is \'realnvp\' or \'style-gan\'.'
    )

    parser.add_argument(
        '--num-hidden',
        type = int,
        default = 512,
        help = 'Number of hidden channels in the ActNorm layer (used for convolutions within the layer as in the Glow paper). Default is 512. Ignored when \'--algo\' argument is \'real-nvp\' or \'style-gan\'.'
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
        '--warmup',
        type = int,
        default = 5,
        help = 'Warmup learning rate. By default 5.'
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
        default = work_dir + '/checkpoints',
        help = 'Output directory for the checkpoint information. Default is *current working directory*/\'checkpoints\''
    )

    parser.add_argument(
        '--fresh',
        type = bool,
        default = True,
        help = 'Should the model be trained from the scratch - defaults to true. Only change this if you have trained a model before (all trained models are stored in \'states\' directory).'
    )

    parser.add_argument(
        '--fixed',
        type = bool,
        default = True,
        help = 'Should the seed be fixed for reproducibility - defaults to true and is set to 999.'
    )

    args = vars(parser.parse_args())

    main(**args)
