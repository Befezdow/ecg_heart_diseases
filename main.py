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
    models = [
        {
            'name': 'Torch neural network',
            'classifier': TorchNetClassifier(attrs_count=abductions_count * needed_per_abduction + 2, classes_count=9),
        },
        {
            'name': 'Random forest',
            'classifier': RFClassifier(estimators_number=110),
        },
    ]

    for epoch in range(epoch_number):
        data = make_augmentation(raw_data, multiplier, abductions_count, needed_per_abduction, base_columns_names)
        data = normalize_and_vectorize(data, class_column_name, excess_column_names)
        splitted_data = split_data(data, class_column_name, test_set_size, random_state)

        train_x = splitted_data['train_set']['x']
        train_y = splitted_data['train_set']['y']
        test_x = splitted_data['test_set']['x']
        test_y = splitted_data['test_set']['y']

        print(f'Epoch: {epoch}')
        for model in models:
            name = model['name']
            classifier = model['classifier']

            train_error = classifier.fit(train_x, train_y)
            test_error = classifier.check(test_x, test_y)

            print(f'Name: {name}; Train error: {train_error}; Test error: {test_error}')
