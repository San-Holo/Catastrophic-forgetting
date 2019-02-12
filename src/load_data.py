import torch
from torchvision import transforms, datasets

from cutout.util.cutout import Cutout


def load_dataset(dataset, path_dir_data, data_augmentation, cutout, n_holes,
                 length_holes):
    if dataset == "mnist":
        return load_mnist(path_dir_data)
    elif dataset == "permuted_mnist":
        return load_permuted_mnist(path_dir_data)
    elif dataset == 'cifar10':
        return load_cifar10(path_dir_data, data_augmentation, cutout, n_holes,
                            length_holes)
    elif dataset == 'cifar100':
        return load_cifar10(path_dir_data, data_augmentation, cutout, n_holes,
                            length_holes)


def load_mnist(path_dir_data):
    num_classes = 10

    all_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ]
    transform = transforms.Compose(all_transforms)
    trainset = datasets.MNIST(root=path_dir_data, train=True,
                              download=True, transform=transform)
    testset = datasets.MNIST(root=path_dir_data, train=False,
                             download=True, transform=transform)
    return trainset, testset, num_classes


def load_permuted_mnist(path_dir_data):
    num_classes = 10

    pixel_permutation = torch.randperm(28 * 28)
    all_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
        transforms.Lambda(
            lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28)
        )
    ]
    transform = transforms.Compose(all_transforms)
    trainset = datasets.MNIST(root=path_dir_data, train=True, download=True,
                              transform=transform)
    testset = datasets.MNIST(root=path_dir_data, train=False, download=True,
                             transform=transform)
    return trainset, testset, num_classes


def load_cifar10(path_dir_data, data_augmentation, cutout, n_holes,
                 length_holes):
    num_classes = 10

    # Image Preprocessing
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )

    # Train transforms
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

    # Test transforms
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = datasets.CIFAR10(root=path_dir_data, train=True,
                                transform=train_transform,
                                download=True)
    testset = datasets.CIFAR10(root=path_dir_data, train=False,
                               transform=test_transform,
                               download=True)
    return trainset, testset, num_classes


def load_cifar100(path_dir_data, data_augmentation, cutout, n_holes,
                  length_holes):
    num_classes = 100

    # Image Preprocessing
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )

    # Train transforms
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

    # Test transforms
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    trainset = datasets.CIFAR100(root=path_dir_data, train=True,
                                 transform=train_transform,
                                 download=True)
    testset = datasets.CIFAR100(root=path_dir_data, train=False,
                                transform=test_transform,
                                download=True)
    return trainset, testset, num_classes


if __name__ == '__main__':
    pass
