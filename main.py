import torch
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

from classifiers import ConvNN, FullyConnectedNN
from data_manager import DataManager
from dataset import check_classes_balance, check_gender_age_stats, get_mean_deviation
from explainable_nn import ExplainableNN


def log_statistics(writer, epoch_number, index, dataset_size, train_loss, train_accuracy, test_loss, test_accuracy):
    print(
        f'[TIME]: Epoch: {epoch_number}, Index: {index} \n'
        f'[TRAIN]: Loss: {train_loss} , Accuracy: {train_accuracy} \n'
        f'[TEST]: Loss: {test_loss} , Accuracy: {test_accuracy} \n'
        f'-----------------------------------------\n'
    )

    position = epoch_number * dataset_size + index
    writer.add_scalar('Train/Loss', train_loss, position)
    writer.add_scalar('Train/Acc', train_accuracy, position)
    writer.add_scalar('Test/Loss', test_loss, position)
    writer.add_scalar('Test/Acc', test_accuracy, position)


def train_net(model, data_manager, epochs=20):
    def test_net(_model, _criterion, _loader):
        _model.eval()

        with torch.no_grad():
            test_loss = 0
            test_accuracy = 0
            for [test_x1, test_x2], test_y in _loader:
                if torch.cuda.is_available():
                    test_x1, test_x2, test_y = test_x1.cuda(), test_x2.cuda(), test_y.cuda()

                test_out = _model(test_x1, test_x2)
                test_loss = _criterion(test_out, test_y)
                _, test_pred = torch.max(test_out.data, 1)

                test_loss += test_loss.item()
                test_accuracy += test_pred.eq(test_y).sum().item() / test_y.size(0)

            test_dataset_size = len(_loader)
            test_loss /= test_dataset_size
            test_accuracy /= test_dataset_size

        _model.train()
        return test_loss, test_accuracy

    writer = SummaryWriter(f'./logs/ConvNN-{datetime.datetime.now()}')

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    train_loader, test_loader = data_manager.get_cnn_train_test_loaders()
    print(f'Train dataset size: {len(train_loader.dataset)}; Test dataset size: {len(test_loader.dataset)}')

    model.train()
    print(f'Started learn ConvNN {datetime.datetime.now()}')

    for epoch_number in range(epochs):
        for index, ([train_x1, train_x2], train_y) in enumerate(train_loader):
            if torch.cuda.is_available():
                train_x1, train_x2,  train_y = train_x1.cuda(), train_x2.cuda(), train_y.cuda()

            optimizer.zero_grad()

            train_out = model(train_x1, train_x2)
            print(train_out)
            train_loss = criterion(train_out, train_y)
            _, train_pred = torch.max(train_out.data, 1)

            train_loss.backward()
            optimizer.step()

            train_accuracy = train_pred.eq(train_y).sum().item() / train_y.size(0)

            if index % 1 == 0:
                test_loss, test_accuracy = test_net(model, criterion, test_loader)
                log_statistics(
                    writer, epoch_number, index, len(train_loader), train_loss.item(),
                    train_accuracy, test_loss, test_accuracy
                )


def main():
    marks_csv = 'data/REFERENCE.csv'
    train_dir = 'data/samples'
    print('Classes balance:', check_classes_balance(marks_csv))
    # check_gender_age_stats(marks_csv, train_dir)
    # age_info, data_info = get_mean_deviation(marks_csv, train_dir, intervals=[(1000, 3500)])
    # print(age_info)
    # print(data_info)

    data_manager = DataManager(marks_csv, train_dir, batch_size=2048, augment_multiplier=10)
    # network = ConvNN()
    # network = FullyConnectedNN(500)
    network = ExplainableNN()
    train_net(network, data_manager)


if __name__ == '__main__':
    main()
