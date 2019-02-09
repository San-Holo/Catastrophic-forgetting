import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """ Basic CNN model. """

    def __init__(self, dropout=True):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        if dropout:
            self.dropout = True
            self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        if self.dropout:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))

        if self.dropout:
            x = F.dropout(x, training=self.training)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    pass
