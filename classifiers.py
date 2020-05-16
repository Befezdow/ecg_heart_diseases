from torch import nn
from torch.nn import functional as F


class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(3, 9, kernel_size=(1, 3))
        self.conv3 = nn.Conv2d(9, 27, kernel_size=(1, 3))

        # max pooling
        self.maxPooling = nn.MaxPool2d(kernel_size=(1, 3))

        self.feed_forward_input_size = 27 * 12 * 91

        # feed forward
        self.fc1 = nn.Linear(self.feed_forward_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 9)

        # dropouts
        self.dropout50 = nn.Dropout(0.5)
        self.dropout25 = nn.Dropout(0.25)

    def forward(self, x):
        print(x.shape)
        x = self.maxPooling(F.relu(self.conv1(x)))
        print(x.shape)
        x = self.maxPooling(F.relu(self.conv2(x)))
        print(x.shape)
        x = self.maxPooling(F.relu(self.conv3(x)))
        print(x.shape)

        x = self.dropout50(x)
        x = x.view(-1, self.feed_forward_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout50(x)
        x = F.relu(self.fc2(x))
        x = self.dropout25(x)
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
