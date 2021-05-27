import torch
import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from cam import draw_cam
from config import get_model, get_data_manager, get_cam_extractor, training_epochs_count


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


def calculate_confusion_matrix(model, data_manager):
    if torch.cuda.is_available():
        model.cuda()

    model.eval()

    sources = []
    predictions = []
    with torch.no_grad():
        test_loader = data_manager.get_test_loader(need_shuffle=True)
        for index, (test_x1, test_x2, test_y) in enumerate(test_loader):
            if torch.cuda.is_available():
                test_x1, test_x2, test_y = test_x1.cuda(), test_x2.cuda(), test_y.cuda()

            test_out = model(test_x1, test_x2)
            _, test_pred = torch.max(test_out.data, 1)

            sources += test_y.cpu().numpy().tolist()
            predictions += test_pred.cpu().numpy().tolist()

    cm = confusion_matrix(sources, predictions)

    print("Confusion matrix:")
    print(cm)


def fetch_and_draw_cam(model, data_manager, records_to_show=10, need_shuffle=True):
    model.eval()

    extract_func = get_cam_extractor()

    iterator = iter(data_manager.get_test_loader(need_shuffle=need_shuffle, custom_batch_size=1))
    for i in range(0, records_to_show):
        sample = next(iterator)
        (x1, x2, y) = sample
        print(f'Original class: {y}')
        cam = extract_func(model, sample)
        draw_cam(sample, cam)


def main():
    data_manager = get_data_manager()
    model = get_model()

    # training
    train_net(model, data_manager, epochs=training_epochs_count)

    # confusion matrix
    calculate_confusion_matrix(model, data_manager)

    # CAM extraction
    fetch_and_draw_cam(model, data_manager)


if __name__ == '__main__':
    print(f'Is CUDA available: {torch.cuda.is_available()}')
    main()
