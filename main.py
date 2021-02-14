import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from scipy import interpolate
from matplotlib import pyplot as plt
from matplotlib import cm

from cam import extract_cam
from data_manager import DataManager
# from dataset import check_classes_balance, check_gender_age_stats, get_mean_deviation
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
            for test_x1, test_x2, test_y in _loader:
                if torch.cuda.is_available():
                    test_x1, test_x2, test_y = test_x1.cuda(), test_x2.cuda(), test_y.cuda()

                test_out = _model(test_x1, test_x2)
                test_loss = _criterion(test_out, test_y.long())
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

    train_loader = data_manager.get_train_loader()
    test_loader = data_manager.get_test_loader()

    print(f'Learning started at {datetime.datetime.now()}')
    print(f'Train dataset size: {len(train_loader.dataset)}')
    print(f'Test dataset size: {len(test_loader.dataset)}')

    model.train()
    for epoch_number in range(epochs):
        for index, (train_x1, train_x2, train_y) in enumerate(train_loader):
            if torch.cuda.is_available():
                train_x1, train_x2, train_y = train_x1.cuda(), train_x2.cuda(), train_y.cuda()

            optimizer.zero_grad()

            train_out = model(train_x1, train_x2)
            train_loss = criterion(train_out, train_y.long())
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
    # print('Classes balance:', check_classes_balance(marks_csv))
    # check_gender_age_stats(marks_csv, train_dir)
    # age_info, data_info = get_mean_deviation(marks_csv, train_dir, intervals=[(1000, 3500)])
    # print(age_info)
    # print(data_info)

    data_manager = DataManager(
        train_dir='data/train',
        test_dir='data/test',
        labels_file='labels.csv',
        data_folder='samples',
        batch_size=1
    )

    # network = ConvNN()
    # network = FullyConnectedNN(500)
    model = ExplainableNN()

    # train_net(model, data_manager)

    sample = next(iter(data_manager.get_test_loader(need_shuffle=False, custom_batch_size=1)))
    cam = extract_cam(model, 'dropout_9', 'linear', sample)[0]

    def draw_heat_chart(data_sample, data_cam):
        sample_timeseries_length = data_sample[1].shape[2]

        x_values = list(range(0, sample_timeseries_length))
        y_values = data_sample[1][0, 0].tolist()  # TODO for each channel
        heat_values = []
        for x_value in x_values:
            cam_index = int(round(x_value / (sample_timeseries_length - 1) * (data_cam.shape[0] - 1)))
            heat_values.append(data_cam[cam_index])

        heat_func = interpolate.interp1d(x_values, heat_values, copy=False)
        plt.scatter(x_values, y_values, c=cm.hot(heat_func(x_values)), edgecolor='none')

    draw_heat_chart(sample, cam)


if __name__ == '__main__':
    main()
