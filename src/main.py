import time
import os
import pickle
import random

import click
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import MultiStepLR

from cutout.util.cutout import Cutout
from cutout.model.resnet import ResNet18
from cutout.model.wide_resnet import WideResNet
from models import CNN
from utils import get_hms, noisy, train, test


@click.command()
@click.argument("dataset", type=click.Choice(['mnist', 'permuted_mnist',
                                              'cifar10', 'cifar100']))
@click.argument("model", type=click.Choice(['cnn', 'resnet18', 'wideresnet']))
@click.option('--batch-size', type=int, default=128,
              help='Input batch size for training')
@click.option('--epochs', type=int, default=200,
              help='Number of epochs to train')
@click.option('--optimizer', type=click.Choice(['sgd', 'adam']), default='sgd',
              help='Optimizer to use')
@click.option('--learning-rate', type=float, default=0.1,
              help='Learning rate')
@click.option('--momentum', type=float, default=0.5,
              help='SGD momentum')
@click.option('--cuda', is_flag=True,
              help='Enables CUDA training')
@click.option('--no-dropout', is_flag=True,
              help='Disable dropout')
@click.option('--data-augmentation', is_flag=True,
              help='Augment data by flipping and cropping')
@click.option('--cutout', is_flag=True,
              help='Apply cutout')
@click.option('--n-holes', type=int, default=1,
              help='Number of holes to cut out from image with cutout')
@click.option('--length-holes', type=int, default=16,
              help='Length of the holes with cutout')
@click.option('--seed', type=int, default=0,
              help='Set the random seed')
@click.option('--path-dir-data', type=str, default="../data",
              help='Path to the dir containing the data')
@click.option('--input-dir', type=str, default="cifar10_results",
              help='Set the random seed')
@click.option('--output-dir', type=str, default="output",
              help='Set the random seed')
# TODO: Comprendre ces arguments
@click.option('--sorting-file', type=str, default="none",
              help=('name of a file containing order of examples sorted by for'
                    'getting'))
@click.option('--remove-n', type=int, default=0,
              help='number of sorted examples to remove from training')
@click.option('--keep-lowest-n', type=int, default=0,
              help=('number of sorted examples to keep that have the lowest sc'
                    'ore, equivalent to start index of removal, if a negative '
                    'number given, remove random draw of examples'))
@click.option('--remove-subsample', type=int, default=0,
              help=('number of examples to remove from the keep-lowest-n examp'
                    'les'))
@click.option('--noise-percent-labels', type=int, default=0,
              help='percent of labels to randomly flip to a different label')
@click.option('--noise-percent-pixels', type=int, default=0,
              help='percent of pixels to randomly introduce Gaussian noise to')
@click.option('--noise-std-pixels', type=float, default=0,
              help='standard deviation of Gaussian pixel noise')
def main(dataset, model, batch_size, epochs, optimizer, learning_rate,
         momentum, cuda, no_dropout, data_augmentation, cutout, n_holes,
         length_holes, seed, path_dir_data, input_dir, output_dir,
         sorting_file, remove_n, keep_lowest_n, remove_subsample,
         noise_percent_labels, noise_percent_pixels, noise_std_pixels):
    # Process output filename
    args = locals()
    ordered_args = [
        'dataset', 'data_augmentation', 'cutout', 'seed', 'sorting_file',
        'remove_n', 'keep_lowest_n', 'remove_subsample',
        'noise_percent_labels', 'noise_percent_pixels', 'noise_std_pixels'
    ]
    save_fname = '__'.join('{}_{}'.format(arg, args[arg])
                           for arg in ordered_args)

    # Set appropriate devices
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    cudnn.benchmark = True  # Should make training go faster for large models

    # Set random seed for initialization
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Load the appropriate train and test datasets
    if dataset == "mnist":
        all_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]
        transform = transforms.Compose(all_transforms)
        trainset = datasets.MNIST(root=path_dir_data, train=True,
                                  download=True, transform=transform)
        testset = datasets.MNIST(root=path_dir_data, train=False,
                                 download=True, transform=transform)
    elif dataset == "permuted_mnist":
        pixel_permutation = torch.randperm(28 * 28)
        all_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, )),
            transforms.Lambda(
                lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28)
            )
        ]
        transform = transforms.Compose(all_transforms)
        trainset = datasets.MNIST(root=path_dir_data, train=True,
                                  download=True, transform=transform)
        testset = datasets.MNIST(root=path_dir_data, train=False,
                                 download=True, transform=transform)
    elif dataset == 'cifar10':
        # Image Preprocessing
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )

        # Setup train transforms
        train_transform = transforms.Compose([])
        if data_augmentation:
            train_transform.transforms.append(
                transforms.RandomCrop(32, padding=4)
            )
            train_transform.transforms.append(
                transforms.RandomHorizontalFlip()
            )
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        if cutout:
            train_transform.transforms.append(
                Cutout(n_holes=n_holes, length=length_holes)
            )

        # Setup test transforms
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        num_classes = 10
        trainset = datasets.CIFAR10(root=path_dir_data, train=True,
                                    transform=train_transform,
                                    download=True)
        testset = datasets.CIFAR10(root=path_dir_data, train=False,
                                   transform=test_transform,
                                   download=True)
    elif dataset == 'cifar100':
        # Image Preprocessing
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )

        # Setup train transforms
        train_transform = transforms.Compose([])
        if data_augmentation:
            train_transform.transforms.append(
                transforms.RandomCrop(32, padding=4)
            )
            train_transform.transforms.append(
                transforms.RandomHorizontalFlip()
            )
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        if cutout:
            train_transform.transforms.append(
                Cutout(n_holes=n_holes, length=length_holes)
            )

        # Setup test transforms
        test_transform = transforms.Compose([transforms.ToTensor(), normalize])

        num_classes = 100
        trainset = datasets.CIFAR100(root=path_dir_data, train=True,
                                     transform=train_transform,
                                     download=True)
        testset = datasets.CIFAR100(root=path_dir_data, train=False,
                                    transform=test_transform,
                                    download=True)

    os.makedirs(output_dir, exist_ok=True)

    # Get indices of examples that should be used for training
    if sorting_file == 'none':
        train_indx = np.array(range(len(trainset.train_labels)))
    else:
        try:
            with open(os.path.join(input_dir, sorting_file) + '.pkl', 'rb') \
                    as fin:
                ordered_indx = pickle.load(fin)['indices']
        except IOError:
            with open(os.path.join(input_dir, sorting_file), 'rb') as fin:
                ordered_indx = pickle.load(fin)['indices']

        # Get the indices to remove from training
        elements_to_remove = np.array(
            ordered_indx)[keep_lowest_n:keep_lowest_n + remove_n]

        # Remove the corresponding elements
        train_indx = np.setdiff1d(range(len(trainset.train_labels)),
                                  elements_to_remove)

    # Remove remove_n number of examples from the train set at random
    if keep_lowest_n < 0:
        train_indx = np.random.permutation(np.arange(len(
            trainset.train_labels)))[:len(trainset.train_labels) - remove_n]

    # Remove remove_sample number of examples at random from the first
    # keep_lowest_n examples. Useful when the first keep_lowest_n examples have
    # equal forgetting counts
    elif remove_subsample:
        lowest_n = np.array(ordered_indx)[0:keep_lowest_n]
        train_indx = lowest_n[np.random.permutation(np.arange(
            keep_lowest_n))[:keep_lowest_n - remove_subsample]]
        train_indx = np.hstack((train_indx,
                                np.array(ordered_indx)[keep_lowest_n:]))

    # Reassign train data and labels
    trainset.train_data = trainset.train_data[train_indx, :, :]
    trainset.train_labels = \
        np.array(trainset.train_labels)[train_indx].tolist()

    # Introduce noise to images if specified
    if noise_percent_pixels:
        for ind in range(len(train_indx)):
            image = trainset.train_data[ind, :, :, :]
            noisy_image = noisy(image, noise_percent_pixels,
                                noise_std_pixels)
            trainset.train_data[ind, :, :, :] = noisy_image

    # Introduce noise to labels if specified
    if noise_percent_labels:
        fname = os.path.join(output_dir, save_fname)

        with open(fname + "_changed_labels.txt", "w") as f:
            # Compute number of labels to change
            nlabels = len(trainset.train_labels)
            nlabels_to_change = int(noise_percent_labels * nlabels / 100)
            nclasses = len(np.unique(trainset.train_labels))
            print('flipping ' + str(nlabels_to_change) + ' labels')

            # Randomly choose which labels to change, get indices
            labels_inds_to_change = np.random.choice(
                np.arange(nlabels), nlabels_to_change, replace=False)

            # Flip each of the randomly chosen labels
            for l, label_ind_to_change in enumerate(labels_inds_to_change):
                # Possible choices for new label
                label_choices = np.arange(nclasses)

                # Get true label to remove it from the choices
                true_label = trainset.train_labels[label_ind_to_change]

                # Remove true label from choices
                label_choices = np.delete(
                    label_choices,
                    true_label)  # the label is the same as the index of the label

                # Get new label and relabel the example with it
                noisy_label = np.random.choice(label_choices, 1)
                trainset.train_labels[label_ind_to_change] = noisy_label[0]

                # Write the example index from the original example order, the
                # old, and the new label
                f.write(
                    str(train_indx[label_ind_to_change]) + ' ' +
                    str(true_label) + ' ' + str(noisy_label[0]) + '\n')

    print('Training on ' + str(len(trainset.train_labels)) + ' examples')

    # Setup model
    if model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif model == 'wideresnet':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                           dropRate=0.3)
    elif model == "cnn":
        model = CNN()
    model = model.to(device)

    # Setup loss
    criterion = nn.CrossEntropyLoss().cuda()
    criterion.__init__(reduction='none')

    # Setup optimizer
    if optimizer == 'adam':
        model_optimizer = torch.optim.Adam(model.parameters(),
                                           lr=learning_rate)
    elif optimizer == 'sgd':
        model_optimizer = torch.optim.SGD(model.parameters(),
                                          lr=learning_rate,
                                          momentum=momentum,
                                          nesterov=True,
                                          weight_decay=5e-4)
        scheduler = MultiStepLR(model_optimizer, milestones=[60, 120, 160],
                                gamma=0.2)

    # Initialize dictionary to save statistics for every example presentation
    example_stats = {}

    best_acc = 0
    elapsed_time = 0
    for epoch in range(epochs):
        start_time = time.time()

        train(trainset, model, model_optimizer, criterion, batch_size, device,
              epoch, example_stats, epochs, train_indx)
        test(testset, model, criterion, device, example_stats, epoch,
             output_dir, dataset, save_fname, best_acc)

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

        # Update optimizer step
        if optimizer == 'sgd':
            scheduler.step(epoch)

        # Save the stats dictionary
        fname = os.path.join(output_dir, save_fname)
        with open(fname + "__stats_dict.pkl", "wb") as f:
            pickle.dump(example_stats, f)

        # Log the best train and test accuracy so far
        with open(fname + "__best_acc.txt", "w") as f:
            f.write('train test \n')
            f.write(str(max(example_stats['train'][1])))
            f.write(' ')
            f.write(str(max(example_stats['test'][1])))


if __name__ == '__main__':
    main()
