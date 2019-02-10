import torch.nn as nn


def load_criterion(criterion, device):
    if criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss().cuda()
        criterion.__init__(reduction='none')

    return criterion.to(device)


if __name__ == '__main__':
    pass
