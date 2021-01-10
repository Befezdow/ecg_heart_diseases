import json
import os
import csv
import scipy.io as sio
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data_file(folder_path, file_name, extension):
    short_data_filename = f'{file_name}.{extension}'
    full_data_filename = os.path.join(folder_path, short_data_filename)
    raw_data = sio.loadmat(full_data_filename)

    gender = raw_data['ECG'][0][0][0][0]
    gender = 0 if gender == 'Male' else 1  # vectorization

    age = raw_data['ECG'][0][0][1][0][0]
    data = raw_data['ECG'][0][0][2].astype(float)

    additional_data = [
        gender,  # gender
        age,  # age
        0,  # height
        0,  # weight
    ]
    return data, additional_data


def split_data(labels_file, id_column, label_column, test_part=0.2):
    labels_frame = pd.read_csv(labels_file)
    labels_frame = labels_frame[[id_column, label_column]]  # дропаем лишние колонки

    unique_labels = labels_frame[label_column].unique()  # получаем уникальные лэйблы

    data_info = {'total_size': 0, 'train_size': 0, 'test_size': 0, 'by_labels': {}}
    train_frame = pd.DataFrame()
    test_frame = pd.DataFrame()
    for label in unique_labels:  # делаем разбиение на samples и test для каждого класса отдельно
        labelled_rows = labels_frame[labels_frame[label_column] == label]
        labelled_train, labelled_test = train_test_split(labelled_rows, test_size=test_part)
        train_frame = pd.concat([train_frame, labelled_train], axis=0, sort=False, ignore_index=True)
        test_frame = pd.concat([test_frame, labelled_test], axis=0, sort=False, ignore_index=True)

        data_info['total_size'] += labelled_rows.size.item()
        data_info['train_size'] += train_frame.size.item()
        data_info['test_size'] += test_frame.size.item()
        data_info['by_labels'][label.item()] = {
            'total_size': labelled_rows.size.item(),
            'train_size': train_frame.size.item(),
            'test_size': test_frame.size.item(),
        }

    return (train_frame, test_frame), data_info


def preprocess_data(
        input_labels_file,
        id_column_name,
        label_column_name,
        split_test_part=0.2,
        input_data_folder='data/samples',
        input_data_extension='mat',
):
    output_labels_file = 'labels.csv'
    train_folder_name = 'data/train'
    test_folder_name = 'data/test'

    def preprocess_frame(frame, input_folder, input_extension, output_folder, output_labels_file):
        current_output_id = 0
        full_output_labels_file = os.path.join(output_folder, output_labels_file)
        samples_folder = os.path.join(output_folder, 'samples')

        if not os.path.exists(output_folder):
            print(f"PREPROCESS_DATA :: Train folder doesn't exist. Creating folder {output_folder} ...")
            os.makedirs(output_folder)
            os.makedirs(samples_folder)
        else:
            if os.path.exists(full_output_labels_file) and os.path.exists(samples_folder):
                labels_frame = pd.read_csv(full_output_labels_file)
                last_row_id = int(labels_frame.tail(1).to_numpy()[0, 0])
                current_output_id = last_row_id + 1
                print(f"PREPROCESS_DATA :: {output_folder} folder detected. Start output id: {current_output_id}")
            else:
                print(f"PREPROCESS_DATA :: {output_folder} folder detected but it has incorrect format. Interrupting ...")
                exit(1)

        with open(full_output_labels_file, 'a') as labels_file:
            writer = csv.writer(labels_file)
            for index, row in frame.iterrows():
                row_id = row[id_column_name]
                row_label = row[label_column_name]
                row_data, row_additional_data = load_data_file(input_folder, row_id, input_extension)

                # TODO do augmentation or zero-padding to row_data

                writer.writerow([current_output_id, row_label] + row_additional_data)

                full_row_data_filename = os.path.join(output_folder, 'samples', f'{current_output_id}.csv')
                pd.DataFrame(row_data.transpose()).to_csv(full_row_data_filename, header=False, index=False)

                current_output_id += 1

                if index % 100 == 0:
                    print(f'Progress: {index}/{frame.size} - {round(index / frame.size * 100, 2)}%')

    (train_frame, test_frame), data_info = split_data(input_labels_file, id_column_name, label_column_name, split_test_part)

    print(f'PREPROCESS_DATA :: Dataset information: \n {json.dumps(data_info, indent=2)}')

    print(f'PREPROCESS_DATA :: Train dataset preprocessing...')
    preprocess_frame(train_frame, input_data_folder, input_data_extension, train_folder_name, output_labels_file)

    print(f'PREPROCESS_DATA :: Test dataset preprocessing...')
    preprocess_frame(test_frame, input_data_folder, input_data_extension, test_folder_name, output_labels_file)


if __name__ == '__main__':
    preprocess_data('data/REFERENCE.csv', 'Recording', 'First_label', 0.2, 'data/samples', 'mat')