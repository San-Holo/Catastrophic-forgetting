import time
import os
import pickle

import torch.optim as optim

from torchvision import datasets, transforms


# Enter all arguments that you want to be in the filename of the saved output
ordered_args = [
    'dataset', 'no_dropout', 'seed', 'sorting_file', 'remove_n',
    'keep_lowest_n'
]

# Parse arguments and setup name of output file with forgetting stats
args = parser.parse_args()
args_dict = vars(args)
print(args_dict)
save_fname = '__'.join(
    '{}_{}'.format(arg, args_dict[arg]) for arg in ordered_args)

# Set appropriate devices
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# Set random seed for initialization
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
npr.seed(args.seed)

# Setup transforms
all_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
]
if args.dataset == 'permuted_mnist':
    pixel_permutation = torch.randperm(28 * 28)
    all_transforms.append(
        transforms.Lambda(
            lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28)))
transform = transforms.Compose(all_transforms)

os.makedirs(args.output_dir, exist_ok=True)

# Load the appropriate train and test datasets
trainset = datasets.MNIST(
    root='/tmp/data', train=True, download=True, transform=transform)
testset = datasets.MNIST(
    root='/tmp/data', train=False, download=True, transform=transform)

# Get indices of examples that should be used for training
if args.sorting_file == 'none':
    train_indx = np.array(range(len(trainset.train_labels)))
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
        range(len(trainset.train_labels)), elements_to_remove)

# Remove remove_n number of examples from the train set at random
if args.keep_lowest_n < 0:
    train_indx = npr.permutation(np.arange(len(
        trainset.train_labels)))[:len(trainset.train_labels) - args.remove_n]

# Reassign train data and labels
trainset.train_data = trainset.train_data[train_indx, :, :]
trainset.train_labels = np.array(trainset.train_labels)[train_indx].tolist()

print('Training on ' + str(len(trainset.train_labels)) + ' examples')

# Setup model and optimizer
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Setup loss
criterion = nn.CrossEntropyLoss()
criterion.__init__(reduce=False)

# Initialize dictionary to save statistics for every example presentation
example_stats = {}

elapsed_time = 0
for epoch in range(args.epochs):
    start_time = time.time()

    train(args, model, device, trainset, optimizer, epoch, example_stats)
    test(args, model, device, testset, example_stats)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

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
