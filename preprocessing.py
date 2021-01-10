import json
import math
import os
import csv

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

        data_info['total_size'] += labelled_rows.shape[0]  # собираем информацию по датасету
        data_info['train_size'] += labelled_train.shape[0]
        data_info['test_size'] += labelled_test.shape[0]
        data_info['by_labels'][label.item()] = {
            'total_size': labelled_rows.shape[0],
            'train_size': labelled_train.shape[0],
            'test_size': labelled_test.shape[0],
        }

    return (train_frame, test_frame), data_info


def preprocess_data(
        input_labels_file,  # название файла, содержащего метки классов
        id_column_name,  # название колонки, содержащей id записи (название файла)
        label_column_name,  # название колонки, содержащей метки классов
        split_test_part=0.2,  # процент данных в тестовом датасете
        adjusted_augmentation_size=10000,  # итоговая длина таймлайнов элементов данных
        augmentation_overlong_threshold=1000,  # порог, при привышении которого относительно adjusted_augmentation_size срабатывает аугментация
        input_data_folder='data/samples',  # имя папки, содержащей файлы с данными
        input_data_extension='mat',  # расширение файлов с данными
):
    output_labels_file = 'labels.csv'
    train_folder_name = 'data/train'
    test_folder_name = 'data/test'

    def preprocess_frame(
            frame,
            input_folder,
            input_extension,
            output_folder,
            output_labels_file,
            adjusted_augmentation_size,
            augmentation_overlong_threshold,
    ):
        current_output_id = 0
        full_output_labels_file = os.path.join(output_folder, output_labels_file)
        samples_folder = os.path.join(output_folder, 'samples')

        # проверяем существование папки с выходным датасетом
        if not os.path.exists(output_folder):  # если не существует, то создаем
            print(f"PREPROCESS_DATA :: Train folder doesn't exist. Creating folder {output_folder} ...")
            os.makedirs(output_folder)
            os.makedirs(samples_folder)
        else:
            if os.path.exists(full_output_labels_file) and os.path.exists(samples_folder):  # если существует и корректна
                labels_frame = pd.read_csv(full_output_labels_file)
                last_row_id = int(labels_frame.tail(1).to_numpy()[0, 0])
                current_output_id = last_row_id + 1
                print(f"PREPROCESS_DATA :: {output_folder} folder detected. Start output id: {current_output_id}")
            else:  # если существует и некорректна
                print(f"PREPROCESS_DATA :: {output_folder} folder detected but it has incorrect format. Interrupting ...")
                exit(1)

        total_row_data_length = 0
        max_row_data_length = -math.inf
        min_row_data_length = math.inf
        with open(full_output_labels_file, 'a') as labels_file:
            writer = csv.writer(labels_file)
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
                elif row_data_length - adjusted_augmentation_size < augmentation_overlong_threshold:
                    samples_to_save.append(np.delete(row_data, np.s_[adjusted_augmentation_size:], 1))
                # если длина таймлайна больше нужного, и достаточна для аугментации то аугментируем одну запись в три
                else:
                    center_part_offset = math.floor((row_data_length - adjusted_augmentation_size) / 2)

                    # TODO учитывать ситуации, когда размер таймлайна больше чем adjusted_augmentation_size * 3 (макс длина - 72к)
                    samples_to_save.append(row_data[:, 0:adjusted_augmentation_size])  # левая часть таймлайна
                    samples_to_save.append(row_data[:, center_part_offset:center_part_offset+adjusted_augmentation_size])  # центральная часть таймлайна
                    samples_to_save.append(row_data[:, -adjusted_augmentation_size:-1])  # правая часть таймлайна

                for sample in samples_to_save:
                    writer.writerow([current_output_id, row_label] + row_additional_data)
                    full_row_data_filename = os.path.join(output_folder, 'samples', f'{current_output_id}.csv')
                    pd.DataFrame(sample.transpose()).to_csv(full_row_data_filename, header=False, index=False)
                    current_output_id += 1

                if index % 100 == 0:
                    print(f'Progress: {index}/{frame.shape[0]} - {round(index / frame.shape[0] * 100, 2)}%')

        return total_row_data_length / (index + 1), min_row_data_length, max_row_data_length

    (train_frame, test_frame), data_info = split_data(input_labels_file, id_column_name, label_column_name, split_test_part)

    print(f'PREPROCESS_DATA :: Dataset information: \n {json.dumps(data_info, indent=2)}')

    print(f'PREPROCESS_DATA :: Train dataset preprocessing...')
    average_timeline_len, min_timeline_len, max_timeline_len = preprocess_frame(
        train_frame, input_data_folder, input_data_extension, train_folder_name,
        output_labels_file, adjusted_augmentation_size, augmentation_overlong_threshold
    )
    print(f'PREPROCESS_DATA :: TRAIN - Average len: {average_timeline_len}, Minimal len: {min_timeline_len}, Maximal len: {max_timeline_len}')

    print(f'PREPROCESS_DATA :: Test dataset preprocessing...')
    average_timeline_len, min_timeline_len, max_timeline_len = preprocess_frame(
        test_frame, input_data_folder, input_data_extension, test_folder_name,
        output_labels_file, adjusted_augmentation_size, augmentation_overlong_threshold
    )
    print(f'PREPROCESS_DATA :: TEST - Average len: {average_timeline_len}, Minimal len: {min_timeline_len}, Maximal len: {max_timeline_len}')


if __name__ == '__main__':
    preprocess_data('data/REFERENCE.csv', 'Recording', 'First_label', 0.2, 10000, 1000, 'data/samples', 'mat')