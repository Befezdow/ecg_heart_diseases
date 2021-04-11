import json
import math
import os
import csv
import random

import numpy as np
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


def split_data(labels_file, id_column, label_column, test_size=50):
    labels_frame = pd.read_csv(labels_file)
    labels_frame = labels_frame[[id_column, label_column]]  # дропаем лишние колонки

    unique_labels = labels_frame[label_column].unique()  # получаем уникальные лэйблы

    data_info = {'total_size': 0, 'train_size': 0, 'test_size': 0, 'by_labels': {}}
    train_frame = pd.DataFrame()
    test_frame = pd.DataFrame()
    for label in unique_labels:  # делаем разбиение на samples и test для каждого класса отдельно
        labelled_rows = labels_frame[labels_frame[label_column] == label]
        labelled_test = labelled_rows.sample(n=test_size)
        labelled_train = pd.concat([labelled_rows, labelled_test]).drop_duplicates(keep=False)

        train_frame = pd.concat([train_frame, labelled_train], axis=0, sort=False, ignore_index=True)
        test_frame = pd.concat([test_frame, labelled_test], axis=0, sort=False, ignore_index=True)

        data_info['total_size'] += labelled_rows.shape[0]  # собираем информацию по датасету
        data_info['train_size'] += labelled_train.shape[0]
        data_info['test_size'] += labelled_test.shape[0]
        data_info['by_labels'][label.item()] = {
            'total_size': labelled_rows.shape[0],
            'train_size': labelled_train.shape[0],
            'test_size': labelled_test.shape[0],
        }

    return (train_frame, test_frame), data_info


def calculate_class_balance(data_info):
    summary_count = sum(map(lambda elem: elem['total_size'], data_info['by_labels'].values()))
    result = {}
    for key, value in data_info['by_labels'].items():
        result[key] = value['total_size'] / summary_count
    return result


def preprocess_data(
        input_labels_file,  # название файла, содержащего метки классов
        id_column_name,  # название колонки, содержащей id записи (название файла)
        label_column_name,  # название колонки, содержащей метки классов
        split_test_size=50,  # процент данных в тестовом датасете
        adjusted_augmentation_size=10000,  # итоговая длина таймлайнов элементов данных
        augmentation_overlong_threshold=1000,  # порог, при привышении которого относительно adjusted_augmentation_size срабатывает аугментация
        max_augmentation_multiplier=5,  # множитель аугментации (во сколько записей превращается одна)
        input_data_folder='data/samples',  # имя папки, содержащей файлы с данными
        input_data_extension='mat',  # расширение файлов с данными
):
    output_labels_file = 'labels.csv'
    train_folder_name = 'data/train'
    test_folder_name = 'data/test'
    dataset_info_file = 'data/dataset_info.json'

    def preprocess_frame(
            frame,
            input_folder,
            input_extension,
            output_folder,
            output_labels_file,
            need_augmentation,
            adjusted_augmentation_size,
            augmentation_overlong_threshold,
            max_augmentation_multiplier,
            class_balance
    ):
        current_output_id = 0
        full_output_labels_file = os.path.join(output_folder, output_labels_file)
        samples_folder = os.path.join(output_folder, 'samples')

        # проверяем существование папки с выходным датасетом
        if not os.path.exists(output_folder):  # если не существует, то создаем
            print(f"PREPROCESS_DATA :: {output_folder} folder doesn't exist. Creating folder {output_folder} ...")
            os.makedirs(output_folder)
            os.makedirs(samples_folder)
        else:
            if os.path.exists(full_output_labels_file) and os.path.exists(samples_folder):  # если существует и корректна
                labels_frame = pd.read_csv(full_output_labels_file)
                last_row_id = int(labels_frame.tail(1).to_numpy()[0, 0])
                current_output_id = last_row_id + 1
                print(f"PREPROCESS_DATA :: {output_folder} folder detected. Start output id: {current_output_id}")

                answer = ""
                while answer not in ['y', 'n']:
                    answer = input('Append new data to exist [Y/N]?').lower()
                if answer == 'n':
                    exit(0)

            else:  # если существует и некорректна
                print(f"PREPROCESS_DATA :: {output_folder} folder detected but it has incorrect format. Interrupting ...")
                exit(1)

        total_row_data_length = 0
        max_row_data_length = -math.inf
        min_row_data_length = math.inf
        with open(full_output_labels_file, 'a') as labels_file:
            writer = csv.writer(labels_file)

            augmentation_multipliers = {}
            max_part = max(class_balance.values())
            for key, value in class_balance.items():
                augmentation_multipliers[key] = 1 / (value / max_part)

            for index, row in frame.iterrows():
                row_id = row[id_column_name]
                row_label = row[label_column_name]
                row_data, row_additional_data = load_data_file(input_folder, row_id, input_extension)

                # собираем информацию по таймлайнам датасета
                row_data_length = row_data.shape[1]
                total_row_data_length += row_data_length
                if row_data_length > max_row_data_length:
                    max_row_data_length = row_data_length
                elif row_data_length < min_row_data_length:
                    min_row_data_length = row_data_length


                samples_to_save = []
                # если длина таймлайна меньше нужного, то дописываем его слева и справа нулями
                if row_data_length < adjusted_augmentation_size:
                    padding_size = adjusted_augmentation_size - row_data_length
                    left_padding_size = math.floor(padding_size / 2)
                    right_padding_size = padding_size - left_padding_size
                    left_padding = np.zeros((12, left_padding_size), dtype=np.float64)
                    right_padding = np.zeros((12, right_padding_size), dtype=np.float64)
                    samples_to_save.append(np.hstack((left_padding, row_data, right_padding)))
                # если длина таймлайна больше нужного, но недостаточна для аугментации то обрезаем его
                elif not need_augmentation or row_data_length - adjusted_augmentation_size < augmentation_overlong_threshold:
                    samples_to_save.append(np.delete(row_data, np.s_[adjusted_augmentation_size:], 1))
                # если длина таймлайна больше нужного, и достаточна для аугментации то аугментируем одну запись в несколько
                else:
                    # multiplier = math.ceil((row_data_length - adjusted_augmentation_size) / augmentation_overlong_threshold) + 1
                    # multiplier = math.ceil(row_data_length / adjusted_augmentation_size) * augmentation_multiplier

                    multiplier = round(augmentation_multipliers[row_label] * max_augmentation_multiplier)
                    for i in range(0, multiplier):
                        interval_start = random.randint(0, row_data_length - adjusted_augmentation_size)
                        interval_end = interval_start+adjusted_augmentation_size
                        samples_to_save.append(row_data[:, interval_start:interval_end])

                    # простая аугментация на 1 в 3
                    # center_part_offset = math.floor((row_data_length - adjusted_augmentation_size) / 2)
                    # samples_to_save.append(row_data[:, 0:adjusted_augmentation_size])  # левая часть таймлайна
                    # samples_to_save.append(row_data[:, center_part_offset:center_part_offset+adjusted_augmentation_size])  # центральная часть таймлайна
                    # samples_to_save.append(row_data[:, -adjusted_augmentation_size:])  # правая часть таймлайна

                for sample in samples_to_save:
                    writer.writerow([current_output_id, row_label] + row_additional_data)
                    full_row_data_filename = os.path.join(output_folder, 'samples', f'{current_output_id}.csv')
                    pd.DataFrame(sample.transpose()).to_csv(full_row_data_filename, header=False, index=False)
                    current_output_id += 1

                if index % 100 == 0:
                    print(f'Progress: {index}/{frame.shape[0]} - {round(index / frame.shape[0] * 100, 2)}%')

        return total_row_data_length / (index + 1), min_row_data_length, max_row_data_length

    (train_frame, test_frame), data_info = split_data(
        input_labels_file, id_column_name, label_column_name, split_test_size
    )
    class_balance = calculate_class_balance(data_info)
    data_info['class_balance'] = class_balance

    print(f'PREPROCESS_DATA :: Dataset information: \n {json.dumps(data_info, indent=2)}')

    print(f'PREPROCESS_DATA :: Train dataset preprocessing...')
    train_average_timeline_len, train_min_timeline_len, train_max_timeline_len = preprocess_frame(
        frame=train_frame,
        input_folder=input_data_folder,
        input_extension=input_data_extension,
        output_folder=train_folder_name,
        output_labels_file=output_labels_file,
        need_augmentation=True,
        adjusted_augmentation_size=adjusted_augmentation_size,
        augmentation_overlong_threshold=augmentation_overlong_threshold,
        max_augmentation_multiplier=max_augmentation_multiplier,
        class_balance=class_balance
    )
    print(f'PREPROCESS_DATA :: TRAIN - Average len: {train_average_timeline_len}, '
          f'Minimal len: {train_min_timeline_len}, Maximal len: {train_max_timeline_len}')

    print(f'PREPROCESS_DATA :: Test dataset preprocessing...')
    test_average_timeline_len, test_min_timeline_len, test_max_timeline_len = preprocess_frame(
        frame=test_frame,
        input_folder=input_data_folder,
        input_extension=input_data_extension,
        output_folder=test_folder_name,
        output_labels_file=output_labels_file,
        need_augmentation=False,
        adjusted_augmentation_size=adjusted_augmentation_size,
        augmentation_overlong_threshold=augmentation_overlong_threshold,
        max_augmentation_multiplier=max_augmentation_multiplier,
        class_balance=class_balance
    )
    print(f'PREPROCESS_DATA :: TEST - Average len: {test_average_timeline_len}, '
          f'Minimal len: {test_min_timeline_len}, Maximal len: {test_max_timeline_len}')

    with open(dataset_info_file, 'w') as text_file:
        data_info['timeline_length'] = {
            'train_average_timeline_len': train_average_timeline_len,
            'train_min_timeline_len': train_min_timeline_len,
            'train_max_timeline_len': train_max_timeline_len,
            'test_average_timeline_len': test_average_timeline_len,
            'test_min_timeline_len': test_min_timeline_len,
            'test_max_timeline_len': test_max_timeline_len,
        }
        text_file.write(json.dumps(data_info, indent=2))


def check_dataset(labels_file):
    labels_frame = pd.read_csv(labels_file)

    labels_column = labels_frame.iloc[:, 1].astype(str)

    unique_labels = labels_column.unique()  # получаем уникальные лэйблы
    label_counts = labels_column.value_counts()

    result = {
        'count': {},
        'percentage': {}
    }
    total_count = 0
    for label in unique_labels:
        result['count'][label] = int(label_counts[label])
        total_count += int(label_counts[label])

    for key, value in result['count'].items():
        result['percentage'][key] = value / total_count

    return result


if __name__ == '__main__':
    preprocess_data('data/REFERENCE.csv', 'Recording', 'First_label', 50, 5000, 1000, 3, 'data/samples', 'mat')

    train_dataset_data = check_dataset('data/train/labels.csv')
    print('Train result:')
    print(json.dumps(train_dataset_data, indent=2))

    test_dataset_data = check_dataset('data/test/labels.csv')
    print('Test result:')
    print(json.dumps(test_dataset_data, indent=2))
