import torch.nn as nn
import torch.nn.functional as F

from cutout.model.resnet import ResNet18
from cutout.model.wide_resnet import WideResNet


class CNN(nn.Module):
    def __init__(self, dropout=True, dim_hidden=50, kernel_size=5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
        self.fc1 = nn.Linear(320, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, 10)
        if dropout:
            self.dropout = True
            self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        if self.dropout:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = F.relu(self.fc1(x.view(-1, 320)))

        if self.dropout:
            x = F.dropout(x, training=self.training)

        return F.log_softmax(self.fc2(x), dim=1)


def load_model(model, device, num_classes, input_channel):
    if model == 'resnet18':
        model = ResNet18(num_classes=num_classes, input_channel=input_channel)
    elif model == 'wideresnet':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                           dropRate=0.3)
    elif model == "cnn":
        model = CNN()
    return model.to(device)


if __name__ == '__main__':
    pass
