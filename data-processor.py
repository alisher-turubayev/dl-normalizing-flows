import torch

import torchvision.transforms as transforms

from PIL import Image
import argparse

import os

env = os.path.dirname(os.path.abspath(__file__))
final_image_size = (-1, -1)

def save_img_tensors(path, name):
    global env, final_image_size

    filepaths = []

    with os.scandir(path) as iter:
        for entry in iter:
            if not entry.name.startswith('.') and entry.is_file() and os.path.splitext(entry.name)[-1] == '.jpg':
                filepaths += [entry.path]

    if len(filepaths) == 0:
        print('Note: in subdirectory \'{}\', no importable files were found. Images to be imported must be in \'.jpg\' format.'.format(name))
        return

    transform = transforms.Compose([
        transforms.Resize(final_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    image_tensor = torch.zeros(len(filepaths), 3, 64, 64)
    for i, filepath in enumerate(filepaths):
        img = Image.open(filepath)
        image_tensor[i] = transform(img)
    
    try:
        os.makedirs('tensors')
    except OSError:
        pass

    torch.save(image_tensor, os.path.join(env, 'tensors', 'dataset-' + name + '.pt'))

    print('Dataset {0} is saved in \'/tensors/dataset-{0}-.pt\''.format(name))

    return


def main(path, image_size):
    global env, final_image_size

    final_image_size = image_size

    if path == '':
        path = env + '/datasets/'

    with os.scandir(path) as iter:
        for entry in iter:
            if not entry.name.startswith('.') and entry.is_dir():
                save_img_tensors(entry.path, entry.name)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Dataset packer for DL Final Project on Normalizing Flows')

    parser.add_argument(
        '--path',
        type = str,
        default = '',
        help = 'Path to the datasets. By default, the working directory + \'/datasets/\' Note that the path should specify the top-level directory, where each dataset is in a subdirectory.'
    )

    parser.add_argument(
        '--image-size',
        nargs='+',
        type = int,
        default = -1,
        help = 'Size of the images in the transformed dataset. By default, set to 64 x 64.'
    )

    image_size = (64, 64)

    args = vars(parser.parse_args())

    if args['image_size'] != -1 and len(args['image_size']) <= 2:
       image_size = int(args['image_size']) if len(args['image_size']) == 1 else tuple(args['image_size'])
    else:
        print('No argument supplied for ')

    args['image_size'] = image_size

    main(**args)