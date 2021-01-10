from torch import nn
from torch.nn import functional as F


class ExplainableNN(nn.Module):
    def __init__(self):
        super(ExplainableNN, self).__init__()

        input_size = 12
        self.filters_count = 12

        self.conv_layers = []
        self.batch_norm_layers = []
        for i in range(0, self.filters_count):
            self.conv_layers.append(nn.Conv1d(input_size * 2 ** i, input_size * 2 ** (i + 1), kernel_size=3))
            self.batch_norm_layers.append(nn.BatchNorm1d(num_features=input_size * 2 ** (i + 1)))

        self.max_pooling = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout(0.1)

        self.gap = nn.AdaptiveAvgPool1d(12)
        self.linear = nn.Linear(12, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        conv_data = x2
        for i in range(0, self.filters_count):
            conv_data = self.conv_layers[i](conv_data)
            conv_data = self.batch_norm_layers[i](conv_data)
            conv_data = F.relu(conv_data)
            conv_data = self.max_pooling(conv_data)
            conv_data = self.dropout(conv_data)

        conv_data = self.gap(conv_data)
        conv_data = self.linear(conv_data)
        conv_data = self.sigmoid(conv_data)

        return conv_data