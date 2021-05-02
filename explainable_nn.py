import torch
from torch import nn
from torch.nn import functional as F


class ExplainableNN(nn.Module):
    def __init__(self):
        super(ExplainableNN, self).__init__()

        input_size = 12
        self.filters_count = 10

        self.conv_layers = []
        self.batch_norm_layers = []
        max_pooling_filter_numbers = {0, 4, 9}
        for i in range(0, self.filters_count):
            setattr(self, f'conv_{i}', nn.Conv1d(input_size * 2 ** i, input_size * 2 ** (i + 1), kernel_size=3))
            setattr(self, f'batch_norm_{i}', nn.BatchNorm1d(num_features=input_size * 2 ** (i + 1)))
            if i in max_pooling_filter_numbers:
                setattr(self, f'max_pooling_{i}', nn.MaxPool1d(kernel_size=3, stride=2))
            setattr(self, f'dropout_{i}', nn.Dropout(0.1))

        self.gap = nn.AvgPool1d(kernel_size=1241)  # в качестве kernel_size берется размерность канала
        self.linear = nn.Linear(12288, 9)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        conv_data = x2
        for i in range(0, self.filters_count):
            conv_data = getattr(self, f'conv_{i}')(conv_data)
            conv_data = getattr(self, f'batch_norm_{i}')(conv_data)
            conv_data = F.relu(conv_data)

            max_pooling_layer = getattr(self, f'max_pooling_{i}', None)
            if max_pooling_layer is not None:
                conv_data = max_pooling_layer(conv_data)

            conv_data = getattr(self, f'dropout_{i}')(conv_data)

        conv_data = self.gap(conv_data)  # сюда приходит [1, 12288, 1241]
        conv_data = conv_data.view(conv_data.shape[0], -1)
        # data = torch.cat([x1.float(), conv_data], 1)
        data = conv_data
        data = self.linear(data)
        # conv_data = self.sigmoid(conv_data)
        data = F.softmax(data, dim=1)

        return data


class SimpleExplainableNN(nn.Module):
    def __init__(self):
        super(SimpleExplainableNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv1d(12, 24, kernel_size=3)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=3)
        self.conv3 = nn.Conv1d(48, 96, kernel_size=3)

        # max pooling
        self.maxPooling = nn.MaxPool1d(kernel_size=3)

        self.batchNorm = nn.BatchNorm1d(num_features=12)

        # dropouts
        self.dropout25 = nn.Dropout(0.25)

        self.gap = nn.AvgPool1d(kernel_size=184)  # в качестве kernel_size берется размерность канала
        self.linear = nn.Linear(96, 9)

    def forward(self, x1, x2):
        x2 = self.batchNorm(x2)
        x2 = self.maxPooling(F.relu(self.conv1(x2)))
        x2 = self.maxPooling(F.relu(self.conv2(x2)))
        x2 = self.maxPooling(F.relu(self.conv3(x2)))
        x2 = self.dropout25(x2)

        x2 = self.gap(x2)
        x2 = x2.view(x2.shape[0], -1)
        # data = torch.cat([x1.float(), x2], 1)
        data = x2
        data = self.linear(data)
        # conv_data = self.sigmoid(conv_data)
        data = F.softmax(data, dim=1)

        return data


class GradSimpleExplainableNN(nn.Module):
    def __init__(self):
        super(GradSimpleExplainableNN, self).__init__()

        self.conv1 = nn.Conv1d(12, 24, kernel_size=3)
        self.conv2 = nn.Conv1d(24, 48, kernel_size=3)
        self.conv3 = nn.Conv1d(48, 96, kernel_size=3)

        self.fc1 = nn.Linear(self.feed_forward_input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 9)

        self.maxPooling = nn.MaxPool1d(kernel_size=3)

        self.batchNorm = nn.BatchNorm1d(num_features=12)

        self.dropout25 = nn.Dropout(0.25)

        # placeholder for the gradients
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def _apply_conv(self, x):
        x = self.batchNorm(x)
        x = self.maxPooling(F.relu(self.conv1(x)))
        x = self.maxPooling(F.relu(self.conv2(x)))
        x = self.maxPooling(F.relu(self.conv3(x)))
        x = self.dropout25(x)
        return x

    def forward(self, x1, x2):
        x2 = self._apply_conv(x2)

        # register the hook
        h = x2.register_hook(self.activations_hook)

        x2 = x2.view(x2.shape[0], -1)
        x = torch.cat([x1.float(), x2], 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation extraction
    def get_activations(self, x1, x2):
        return self._apply_conv(x2)
