import torch
from torch import nn
from torch.nn import functional as F


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv1d(12, 24, kernel_size=3)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=3)
        self.conv3 = nn.Conv1d(48, 96, kernel_size=3)

        # max pooling
        self.maxPooling = nn.MaxPool1d(kernel_size=3)

        self.batchNorm = nn.BatchNorm1d(num_features=12)

        self.flat_size = 96 * 91
        self.feed_forward_input_size = 96 * 91 + 2

        # feed forward
        self.fc1 = nn.Linear(self.feed_forward_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 9)

        # dropouts
        self.dropout50 = nn.Dropout(0.5)
        self.dropout25 = nn.Dropout(0.25)

    def forward(self, x1, x2):
        x2 = self.batchNorm(x2)
        x2 = self.maxPooling(F.relu(self.conv1(x2)))
        x2 = self.maxPooling(F.relu(self.conv2(x2)))
        x2 = self.maxPooling(F.relu(self.conv3(x2)))
        x2 = self.dropout50(x2)
        x2 = x2.view(-1, self.flat_size)

        x = torch.cat([x1.float(), x2], 1)
        x = F.relu(self.fc1(x))
        x = self.dropout50(x)
        x = F.relu(self.fc2(x))
        x = self.dropout25(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
