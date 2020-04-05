from datetime import datetime
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from preparations import load_and_prepare_data, make_augmentation, normalize_and_vectorize, split_data
from random_forest import RFClassifier
from torch_net import TorchNetClassifier

epoch_number = 10
multiplier = 2
abductions_count = 12
needed_per_abduction = 2500
test_set_size = 0.3
random_state = 3

if __name__ == '__main__':
    filename = 'N:\\Nova\\preview.csv'

    class_column_name = 'class1'
    excess_column_names = ['class2', 'class3']
    base_columns_names = ['gender', 'age', class_column_name, *excess_column_names]
    names_array = [*base_columns_names, *['t{}'.format(i + 1) for i in range(60000)]]

    raw_data = load_and_prepare_data(filename, names_array)
    current_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    models = [
        {
            'name': 'Torch neural network',
            'classifier': TorchNetClassifier(
                attrs_count=abductions_count * needed_per_abduction + 2, classes_count=9, batch_size=1000
            ),
            'writer': SummaryWriter(f'./logs/torch_net_{current_time}'),
            'batch_size': 1000,
        },
        {
            'name': 'Random forest',
            'classifier': RFClassifier(estimators_number=110, batch_size=1000),
            'writer': SummaryWriter(f'./logs/rf_{current_time}'),
            'batch_size': 1000,
        },
    ]

    for epoch in range(epoch_number):
        data = make_augmentation(raw_data, multiplier, abductions_count, needed_per_abduction, base_columns_names)
        data = normalize_and_vectorize(data, class_column_name, excess_column_names)
        splitted_data = split_data(data, class_column_name, test_set_size, random_state)

        train_x = splitted_data['train_set']['x']
        train_y = splitted_data['train_set']['y']
        train_size = splitted_data['train_set']['size']

        test_x = splitted_data['test_set']['x']
        test_y = splitted_data['test_set']['y']

        print(f'Epoch: {epoch}')
        for model in models:
            name = model['name']
            classifier = model['classifier']

            batches_count = int(np.ceil(train_size / model['batch_size']))

            def lr_logger(x, y): model['writer'].add_scalar('Train/LR', y, x + epoch * batches_count)

            def loss_logger(x, y): model['writer'].add_scalar('Train/Loss', y, x + epoch * batches_count)

            train_error, train_loss = classifier.fit(train_x, train_y, lr_logger, loss_logger)
            test_error, test_loss = classifier.check(test_x, test_y)

            model['writer'].add_scalar('Train/Error', train_error, epoch)
            model['writer'].add_scalar('Test/Error', test_error, epoch)
            model['writer'].add_scalar('Test/Loss', test_loss, epoch)

            print(f'[Name: {name}]; '
                  f'[Train: error - {train_error}; loss - {train_loss}]; '
                  f'[Test: error - {test_error}; loss - {test_loss}'
            )
