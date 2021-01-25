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
        for i in range(0, self.filters_count):
            setattr(self, f'conv_{i}', nn.Conv1d(input_size * 2 ** i, input_size * 2 ** (i + 1), kernel_size=2))
            setattr(self, f'batch_norm_{i}', nn.BatchNorm1d(num_features=input_size * 2 ** (i + 1)))
            setattr(self, f'max_pooling_{i}', nn.MaxPool1d(kernel_size=2))
            setattr(self, f'dropout_{i}', nn.Dropout(0.1))

        self.gap = nn.AvgPool1d(kernel_size=9)
        # TODO не понятно, какой брать kernel_size у GAP
        # TODO не совпадает размер данных перед GAP и размер весов FC слоя
        # self.gap = nn.AdaptiveAvgPool1d(9)
        self.linear = nn.Linear(442368 + 4, 9)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        conv_data = x2
        for i in range(0, self.filters_count):
            conv_data = getattr(self, f'conv_{i}')(conv_data)
            conv_data = getattr(self, f'batch_norm_{i}')(conv_data)
            conv_data = F.relu(conv_data)
            conv_data = getattr(self, f'max_pooling_{i}')(conv_data)
            conv_data = getattr(self, f'dropout_{i}')(conv_data)

        conv_data = self.gap(conv_data)
        conv_data = conv_data.view(1, -1)
        data = torch.cat([x1.float(), conv_data], 1)
        data = self.linear(data)
        # conv_data = self.sigmoid(conv_data)
        data = F.softmax(data, dim=1)

        return data