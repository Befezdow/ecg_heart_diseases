import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, attrs_count, classes_count):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(attrs_count, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, classes_count)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)


class TorchNetClassifier:
    def __init__(self, attrs_count, classes_count, batch_size=1000):
        # Инициализируем нейронную сеть
        self.model = Net(attrs_count, classes_count).double()
        # Создаем оптимизатор
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        # Создаем функцию потерь
        self.criterion = torch.nn.NLLLoss()

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)

        self.batch_size = batch_size

    def fit(self, x_train, y_train, lr_logger=None, loss_logger=None):
        x_tensor = torch.Tensor(x_train.to_numpy().astype(np.float32))
        y_tensor = torch.Tensor(y_train.to_numpy()).long() - 1

        dataset_size = x_tensor.shape[0]
        batch_size = self.batch_size
        batches_count = int(np.ceil(dataset_size / batch_size))

        correct = 0
        summary_loss = 0
        for batch_index in range(batches_count):
            x_var = Variable(x_tensor[batch_index * batch_size:batch_index * batch_size + batch_size])
            y_var = Variable(y_tensor[batch_index * batch_size:batch_index * batch_size + batch_size])

            self.optimizer.zero_grad()
            net_out = self.model(x_var.double())
            loss = self.criterion(net_out, y_var)

            _, pred = net_out.data.max(1)  # получаем индекс максимального значения

            loss.backward()
            self.optimizer.step()

            correct += pred.eq(y_var.data).sum()
            summary_loss += loss.item()

            if lr_logger is not None:
                lr_value = self.scheduler.get_lr()[0]
                lr_logger(batch_index, lr_value)

            if loss_logger is not None:
                loss_value = loss.item()
                loss_logger(batch_index, loss_value)

        self.scheduler.step()
        return 1 - correct.item() / dataset_size, summary_loss / dataset_size

    def check(self, x, y):
        x_tensor = torch.Tensor(x.to_numpy().astype(np.float32))
        y_tensor = torch.Tensor(y.to_numpy()).long() - 1

        dataset_size = x_tensor.shape[0]
        batch_size = self.batch_size
        batches_count = int(np.ceil(dataset_size / batch_size))

        correct = 0
        summary_loss = 0
        for batch_index in range(batches_count):
            x_var = Variable(x_tensor[batch_index * batch_size:batch_index * batch_size + batch_size])
            y_var = Variable(y_tensor[batch_index * batch_size:batch_index * batch_size + batch_size])

            net_out = self.model(x_var.double())
            loss = self.criterion(net_out, y_var)

            _, pred = net_out.data.max(1)  # получаем индекс максимального значения
            correct += pred.eq(y_var.data).sum()
            summary_loss += loss.item()

        return 1 - correct.item() / dataset_size, summary_loss / dataset_size

    def predict(self, x):
        x_tensor = torch.Tensor(x.to_numpy())
        x_var = Variable(x_tensor)
        return self.model(x_var.double())
