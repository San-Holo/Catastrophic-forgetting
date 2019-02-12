import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def process_output_filename(args):
    ordered_args = [
        'dataset', 'data_augmentation', 'cutout', 'seed', 'sorting_file',
        'remove_n', 'removing_method', 'remove_subsample',
        'noise_percent_labels', 'noise_percent_pixels', 'noise_std_pixels'
    ]
    return '__'.join('{}_{}'.format(arg, args[arg]) for arg in ordered_args)


def set_seeds(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def noisy(image, noise_percentage, noise_std):
    """ Introduce Gaussian noise to noise_percentage of image pixels. """
    row, col, ch = image.shape
    num_corrupt = int(np.floor(noise_percentage * row * col / 100))

    # Randomly choose pixels to add noise to
    xy_coords = np.random.choice(row * col, num_corrupt, replace=False)
    chan_coords = np.random.choice(ch, num_corrupt, replace=True)
    xy_coords = np.unravel_index(xy_coords, (row, col))
    out = np.copy(image)

    # Add randomly generated Gaussian noise to pixels
    for coord in range(num_corrupt):
        noise = np.random.normal(120, noise_std, 1)
        out[xy_coords[0][coord], xy_coords[1][coord],
            chan_coords[coord]] += noise

    return out


if __name__ == '__main__':
    pass
