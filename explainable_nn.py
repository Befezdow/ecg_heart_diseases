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
        conv_data = conv_data.view(1, -1)
        # data = torch.cat([x1.float(), conv_data], 1)
        data = conv_data
        data = self.linear(data)
        # conv_data = self.sigmoid(conv_data)
        data = F.softmax(data, dim=1)

        return data