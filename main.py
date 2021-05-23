import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from cam import extract_cam, draw_cam, extract_grad_cam
from data_manager import DataManager
from explainable_nn import ExplainableNN, SimpleExplainableNN, GradSimpleExplainableNN, RegularizedGradSimpleExplainableNN


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
    def test_net(_model, _criterion, _loader, test_batches_number=3):
        _model.eval()

        with torch.no_grad():
            test_loss = 0
            test_accuracy = 0
            for i in range(0, test_batches_number):
                (test_x1, test_x2, test_y) = next(iter(_loader))
                if torch.cuda.is_available():
                    test_x1, test_x2, test_y = test_x1.cuda(), test_x2.cuda(), test_y.cuda()

                test_out = _model(test_x1, test_x2)
                test_loss = _criterion(test_out, test_y.long())
                _, test_pred = torch.max(test_out.data, 1)

                test_loss += test_loss.item()
                test_accuracy += test_pred.eq(test_y).sum().item() / test_y.size(0)

            test_loss /= test_batches_number
            test_accuracy /= test_batches_number

        _model.train()
        return test_loss, test_accuracy

    current_iso_date = datetime.datetime.now().isoformat()
    writer = SummaryWriter(f'./logs/{type(model).__name__}_{current_iso_date}')

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)

    train_loader = data_manager.get_train_loader()
    test_loader = data_manager.get_test_loader()

    print(f'Learning started at {current_iso_date}')
    print(f'Train dataset size: {len(train_loader.dataset)}')
    print(f'Test dataset size: {len(test_loader.dataset)}')

    # for detecting errors
    # torch.autograd.set_detect_anomaly(True)

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

            if index % 5 == 0:
                print('Calculating test score ...')
                test_loss, test_accuracy = test_net(model, criterion, test_loader)
                log_statistics(
                    writer, epoch_number, index, len(train_loader), train_loss.item(),
                    train_accuracy, test_loss, test_accuracy
                )

        torch.save(model.state_dict(), f'models/{current_iso_date}')


def main():
    data_manager = DataManager(
        train_dir='data/train',
        test_dir='data/test',
        labels_file='labels.csv',
        data_folder='samples',
        batch_size=256
    )

    # model = ExplainableNN()   # CAM-based architecture, too difficult for training on default PC
    # model = SimpleExplainableNN() # CAM-based architecture
    # model = GradSimpleExplainableNN() # Grad-CAM-based architecture
    model = RegularizedGradSimpleExplainableNN()    # Grad-CAM-based architecture + mode dropouts

    # loading already saved model
    # model.load_state_dict(torch.load(f'data/models/2021-02-28T21:10:51.448630'))

    # training
    train_net(model, data_manager, epochs=10)

    model.eval()
    sample = next(iter(data_manager.get_test_loader(need_shuffle=True, custom_batch_size=1)))

    # only for CAM-based architectures
    # cam = extract_cam(model, 'dropout25', 'linear', sample)

    # only for Grad-CAM-based architectures
    cam = extract_grad_cam(model, sample)

    draw_cam(sample, cam)


if __name__ == '__main__':
    print(f'Is CUDA available: {torch.cuda.is_available()}')
    main()
