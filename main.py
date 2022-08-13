# File is a mish-mash from https://github.com/ikostrikov/pytorch-flows/,
#   https://github.com/y0ast/Glow-PyTorch/ and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# 
# TODO: rewrite to 
#   1. normalize inputs, 
#   2. split the dataset into test/validation datasets 
#   3. log-error
#   4. select best performing model after N epochs and use it to generate images

import torch
import torch.optim as optim
import torch.utils.data

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from itertools import islice

from tqdm import tqdm
import argparse

import flows as fnn

import os

def test_mode():
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Epochs to train - just for testing purposes, the value is set to 100
    epochs = 100
    # Limited mode - by default, the algorithms are run with simple architectures to allow for easier testing
    limited_mode = True
    # Set the manual seed to 100 for reproducibility (this is a testing method)
    torch.manual_seed(100)
    # Set default parameters
    datapath = current_path + '/datasets/initial-tests'
    # Dimension to resize the input images to (because images are squarified, one side is sufficient)
    image_size = 64
    # Batch size - set to 64 by default
    batch_size = 64
    # Learning rate
    lr = 5e-4
    # Weight decay
    weight_decay = 5e-5
    # Number of workers for the DataLoader
    num_workers = 2
    # Warmup
    warmup = 5
    # Output directory for model checkpoints
    output_dir = current_path + '/checkpoints'
    
    # Use ImageFolder dataset class
    dataset = dset.ImageFolder(root = datapath, 
        transform=transforms.Compose([
            transforms.Resize(image_size), 
            transforms.CenterCrop(image_size), 
            transforms.ToTensor()
        ])
    )

    # Since the dataset is small (~5 MBs), the dataloader is set to single-processing mode
    # https://pytorch.org/docs/stable/data.html
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = batch_size, 
        shuffle = True, 
        num_workers = num_workers
    )

    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

    # Limited mode check - otherwise, sitting here waiting is torture
    k = 5 
    l = 1
    if not limited_mode:
        k = 32
        l = 3

    model = fnn.Glow(K = k, L = l)
    model = model.to(device)

    optimizer = optim.Adamax(model.parameters(), lr = lr, weight_decay = weight_decay)
    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa

    train(epochs, device, model, dataloader, optimizer, lr_lambda, output_dir, fresh = True)

    try:
        os.makedirs('states')
    except OSError:
        pass
    torch.save(model.state_dict(), current_path + '/states/state_test.pt')
    torch.save(optimizer.state_dict(), current_path + '/states/state_optim_test.pt')

    with torch.no_grad():
        imgs = model.sample(temperature = 0.1).detach().cpu()

        try:
            os.makedirs('images')
        except OSError:
            pass

        torchvision.utils.save_image(imgs, 'images/img_test.png', nrows = 4)

def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses

def train(epochs, device, model, dataloader, optimizer, lr_lambda, output_dir, fresh):
    """model.train()

    pbar = tqdm(total = len(dataloader.dataset))
    for _, data in enumerate(dataloader):
        if isinstance(data, list):
            data = data[0]

        data = data.to(device)
        optimizer.zero_grad()
        model(data)
        optimizer.step()
        pbar.update(data.size(0))

    pbar.close() """

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)

        z, nll, y_logits = model(x, None)
        losses = compute_loss(nll)

        losses["total_loss"].backward()

        optimizer.step()

        return losses

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            _, nll, _ = model(x, None)
            losses = compute_loss(nll, reduction="none")

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

    evaluator = Engine(eval_step)

    Loss(
        lambda x, y: torch.mean(x),
        output_transform=lambda x: (
            x["total_loss"],
            torch.empty(x["total_loss"].shape[0]),
        ),
    ).attach(evaluator, "total_loss")

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    """# load pre-trained model if given
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
            engine.state.iteration = resume_epoch * len(engine.state.dataloader) """

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        """ init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(dataloader, None, 8):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            print(init_batches.shape)
            assert init_batches.shape[0] == 64

            init_targets = None

            model(init_batches, init_targets) """

    #@trainer.on(Events.EPOCH_COMPLETED)
    """ def evaluate(engine):
        evaluator.run(test_loader)

        scheduler.step()
        metrics = evaluator.state.metrics

        losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

        print(f"Validation Results - Epoch: {engine.state.epoch} {losses}") """

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

def main(dataset, algo, is_fresh_start):
    print('Yay, I worked!')
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Normalizing Flows PyTorch Implementation.')

    parser.add_argument(
        '--dataset', 
        type = str, 
        default = 'kaggle-sample', 
        choices = ['kaggle-sample', 'kaggle-full'], 
        help = 'The dataset to be used. Datasets kaggle-sample and kaggle-full were used for testing purposes, but can be utilized for training/evaluation.'
    )

    parser.add_argument(
        '--algo',
        type = str,
        default = 'glow',
        choices = ['glow', 'realnvp', 'style-gan'],
        help = 'The type of algorithm to train.'
    )

    parser.add_argument(
        '--fresh',
        action = argparse.BooleanOptionalAction,
        default = True,
        help = 'Should the model be trained from the scratch - defaults to true. Only change this if you have trained a model before (all trained models are stored in \'states\' directory).'
    )

    parser.add_argument(
        '--test',
        action = argparse.BooleanOptionalAction,
        default = False,
        help = 'Should the program run in test mode - defaults to false. In test mode, \'kaggle-sample\' dataset is used (378 images), all three models are trained, seed is manually set, and hyperparameters are reduced to minimal values (this mode is useful for debugging).'
    )

    args = parser.parse_args()

    if args.test:
        test_mode()
    else:
        kwargs = vars(args)
        del kwargs['test']

        main(**kwargs)
