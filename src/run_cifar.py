import time
import os
import pickle

import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms

from Cutout.util.cutout import Cutout
from Cutout.model.resnet import ResNet18
from Cutout.model.wide_resnet import WideResNet


# Enter all arguments that you want to be in the filename of the saved output
ordered_args = [
    'dataset', 'data_augmentation', 'cutout', 'seed', 'sorting_file',
    'remove_n', 'keep_lowest_n', 'remove_subsample', 'noise_percent_labels',
    'noise_percent_pixels', 'noise_std_pixels'
]

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
save_fname = '__'.join(
    '{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

# Set appropriate devices
args.cuda = not args.no_cuda and torch.cuda.is_available()
use_cuda = args.cuda
device = torch.device("cuda" if use_cuda else "cpu")
cudnn.benchmark = True  # Should make training go faster for large models

# Set random seed for initialization
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
npr.seed(args.seed)

# Image Preprocessing
normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

# Setup train transforms
train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)
if args.cutout:
    train_transform.transforms.append(
        Cutout(n_holes=args.n_holes, length=args.length))

# Setup test transforms
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

os.makedirs(args.output_dir, exist_ok=True)

# Load the appropriate train and test datasets
if args.dataset == 'cifar10':
    num_classes = 10
    train_dataset = datasets.CIFAR10(
        root='/tmp/data/',
        train=True,
        transform=train_transform,
        download=True)

    test_dataset = datasets.CIFAR10(
        root='/tmp/data/',
        train=False,
        transform=test_transform,
        download=True)
elif args.dataset == 'cifar100':
    num_classes = 100
    train_dataset = datasets.CIFAR100(
        root='/tmp/data/',
        train=True,
        transform=train_transform,
        download=True)

    test_dataset = datasets.CIFAR100(
        root='/tmp/data/',
        train=False,
        transform=test_transform,
        download=True)

# Get indices of examples that should be used for training
if args.sorting_file == 'none':
    train_indx = np.array(range(len(train_dataset.train_labels)))
else:
    try:
        with open(
                os.path.join(args.input_dir, args.sorting_file) + '.pkl',
                'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']
    except IOError:
        with open(os.path.join(args.input_dir, args.sorting_file),
                  'rb') as fin:
            ordered_indx = pickle.load(fin)['indices']

    # Get the indices to remove from training
    elements_to_remove = np.array(
        ordered_indx)[args.keep_lowest_n:args.keep_lowest_n + args.remove_n]

    # Remove the corresponding elements
    train_indx = np.setdiff1d(
        range(len(train_dataset.train_labels)), elements_to_remove)

if args.keep_lowest_n < 0:
    # Remove remove_n number of examples from the train set at random
    train_indx = npr.permutation(np.arange(len(
        train_dataset.train_labels)))[:len(train_dataset.train_labels) -
                                      args.remove_n]

elif args.remove_subsample:
    # Remove remove_sample number of examples at random from the first keep_lowest_n examples
    # Useful when the first keep_lowest_n examples have equal forgetting counts
    lowest_n = np.array(ordered_indx)[0:args.keep_lowest_n]
    train_indx = lowest_n[npr.permutation(np.arange(
        args.keep_lowest_n))[:args.keep_lowest_n - args.remove_subsample]]
    train_indx = np.hstack((train_indx,
                            np.array(ordered_indx)[args.keep_lowest_n:]))

# Reassign train data and labels
train_dataset.train_data = train_dataset.train_data[train_indx, :, :, :]
train_dataset.train_labels = np.array(
    train_dataset.train_labels)[train_indx].tolist()

# Introduce noise to images if specified
if args.noise_percent_pixels:
    for ind in range(len(train_indx)):
        image = train_dataset.train_data[ind, :, :, :]
        noisy_image = noisy(image, args.noise_percent_pixels,
                            args.noise_std_pixels)
        train_dataset.train_data[ind, :, :, :] = noisy_image

# Introduce noise to labels if specified
if args.noise_percent_labels:
    fname = os.path.join(args.output_dir, save_fname)

    with open(fname + "_changed_labels.txt", "w") as f:

        # Compute number of labels to change
        nlabels = len(train_dataset.train_labels)
        nlabels_to_change = int(args.noise_percent_labels * nlabels / 100)
        nclasses = len(np.unique(train_dataset.train_labels))
        print('flipping ' + str(nlabels_to_change) + ' labels')

        # Randomly choose which labels to change, get indices
        labels_inds_to_change = npr.choice(
            np.arange(nlabels), nlabels_to_change, replace=False)

        # Flip each of the randomly chosen labels
        for l, label_ind_to_change in enumerate(labels_inds_to_change):

            # Possible choices for new label
            label_choices = np.arange(nclasses)

            # Get true label to remove it from the choices
            true_label = train_dataset.train_labels[label_ind_to_change]

            # Remove true label from choices
            label_choices = np.delete(
                label_choices,
                true_label)  # the label is the same as the index of the label

            # Get new label and relabel the example with it
            noisy_label = npr.choice(label_choices, 1)
            train_dataset.train_labels[label_ind_to_change] = noisy_label[0]

            # Write the example index from the original example order, the old, and the new label
            f.write(
                str(train_indx[label_ind_to_change]) + ' ' + str(true_label) +
                ' ' + str(noisy_label[0]) + '\n')

print('Training on ' + str(len(train_dataset.train_labels)) + ' examples')

# Setup model
if args.model == 'resnet18':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    if args.dataset == 'svhn':
        model = WideResNet(
            depth=16, num_classes=num_classes, widen_factor=8, dropRate=0.4)
    else:
        model = WideResNet(
            depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3)
else:
    print(
        'Specified model not recognized. Options are: resnet18 and wideresnet')

# Setup loss
model = model.cuda()
criterion = nn.CrossEntropyLoss().cuda()
criterion.__init__(reduce=False)

# Setup optimizer
if args.optimizer == 'adam':
    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif args.optimizer == 'sgd':
    model_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4)
    scheduler = MultiStepLR(
        model_optimizer, milestones=[60, 120, 160], gamma=0.2)
else:
    print('Specified optimizer not recognized. Options are: adam and sgd')

# Initialize dictionary to save statistics for every example presentation
example_stats = {}

best_acc = 0
elapsed_time = 0
for epoch in range(args.epochs):
    start_time = time.time()

    train(args, model, device, train_dataset, model_optimizer, epoch,
          example_stats)
    test(epoch, model, device, example_stats)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

    # Update optimizer step
    if args.optimizer == 'sgd':
        scheduler.step(epoch)

    # Save the stats dictionary
    fname = os.path.join(args.output_dir, save_fname)
    with open(fname + "__stats_dict.pkl", "wb") as f:
        pickle.dump(example_stats, f)

    # Log the best train and test accuracy so far
    with open(fname + "__best_acc.txt", "w") as f:
        f.write('train test \n')
        f.write(str(max(example_stats['train'][1])))
        f.write(' ')
        f.write(str(max(example_stats['test'][1])))
