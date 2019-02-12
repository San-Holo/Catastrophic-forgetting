import os
import pickle

import click
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from load_data import load_dataset
from models import load_model
from optimizer import load_optimizer
from criterion import load_criterion
from core import train, test, remove_examples, apply_noise
from utils import set_seeds, process_output_filename


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
@click.option('--criterion', type=click.Choice(['cross_entropy']),
              default='cross_entropy', help='Criterion to use')
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
@click.option('--sorting-file', type=str, default="none",
              help=('name of a file containing order of examples sorted by for'
                    'getting'))
@click.option('--remove-n', type=int, default=0,
              help='number of sorted examples to remove from training')
@click.option('--removing-method', type=click.Choice(["0", "1"]), default="0",
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
def main(dataset, model, batch_size, epochs, optimizer, criterion,
         learning_rate, momentum, cuda, no_dropout, data_augmentation, cutout,
         n_holes, length_holes, seed, path_dir_data, input_dir, output_dir,
         sorting_file, remove_n, removing_method, remove_subsample,
         noise_percent_labels, noise_percent_pixels, noise_std_pixels):
    # Process output filename for saving models
    args = locals()
    save_fname = process_output_filename(args)
    os.makedirs(output_dir, exist_ok=True)

    # Set the appropriate device
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    cudnn.benchmark = True

    # Set random seed for initialization
    set_seeds(seed, cuda=cuda)

    # Load the dataset
    trainset, testset, num_classes = load_dataset(dataset, path_dir_data,
                                                  data_augmentation, cutout,
                                                  n_holes, length_holes)

    # Remove examples from the datasets if specified
    train_indx = remove_examples(trainset, removing_method, remove_n,
                                 remove_subsample, sorting_file, input_dir)

    # Apply noise to data if specified
    apply_noise(trainset, train_indx, noise_percent_pixels,
                noise_percent_labels, noise_std_pixels, output_dir,
                save_fname)

    # Setup model, loss and optimizer
    input_channel = 1 if dataset in ("mnist", "permuted_mnist") else 3
    model = load_model(model, device, num_classes, input_channel)
    criterion = load_criterion(criterion, device)
    model_optimizer = load_optimizer(optimizer, model, learning_rate, momentum)

    # Initialize dictionary to save statistics for every example presentation
    example_stats = {}

    best_acc = 0
    for epoch in tqdm(range(epochs)):
        train(trainset, model, model_optimizer, criterion, batch_size, device,
              epoch, example_stats, epochs, train_indx)
        test(testset, model, criterion, device, example_stats, epoch,
             output_dir, dataset, save_fname, best_acc)

        # Save the stats dictionary
        fname = os.path.join(output_dir, save_fname)
        with open(fname + "__stats_dict.pkl", "wb") as f:
            pickle.dump(example_stats, f)


if __name__ == '__main__':
    main()
