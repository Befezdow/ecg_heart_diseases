import pandas as pd
from sklearn.model_selection import train_test_split
import random


def load_and_prepare_data(filename, names_array):
    data = pd.read_csv(filename, names=names_array)
    return data.fillna(data.median(axis=0), axis=0)


def make_augmentation(data, multiplier, abductions_count, needed_per_abduction, base_columns_names):
    non_sliceable_columns = data.iloc[:, 0:5]
    sliceable_columns = data.iloc[:, 5:]
    sliceable_columns_count = sliceable_columns.shape[1]
    per_abduction = int(sliceable_columns_count / abductions_count)

    new_data = pd.DataFrame()
    for i in range(multiplier):
        random_position = random.randint(0, per_abduction - needed_per_abduction)

        total_columns_slice = pd.DataFrame()
        for j in range(abductions_count):
            start_column = j * per_abduction + random_position
            end_column = j * per_abduction + random_position + needed_per_abduction
            columns_slice = sliceable_columns.iloc[:, start_column: end_column]
            total_columns_slice = pd.concat([total_columns_slice, columns_slice.copy()], axis=1, sort=False,
                                            ignore_index=True)

        completed_slice = pd.concat([non_sliceable_columns.copy(), total_columns_slice], axis=1, sort=False,
                                    ignore_index=True)
        new_data = pd.concat([new_data, completed_slice], axis=0, sort=False, ignore_index=True)

    new_data.columns = [*base_columns_names, *['t{}_{}'.format(i + 1, j + 1) for i in range(abductions_count) for j in
                                               range(needed_per_abduction)]]
    return new_data


def normalize_and_vectorize(data, class_column, excess_columns):
    # убираем лишние классовые признаки
    for excess_column in excess_columns:
        data = data.drop(excess_column, axis=1)

    # собираем числовые колонки
    numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object' and c != class_column]

    # векторизируем колонку gender
    # TODO think how to universalize it
    data.at[data['gender'] == 'Male', 'gender'] = 0
    data.at[data['gender'] == 'Female', 'gender'] = 1

    # нормализуем числовые атрибуты
    data_numerical = data[numerical_columns]
    data_numerical = (data_numerical - data_numerical.mean(axis=0)) / data_numerical.std(axis=0)
    data[numerical_columns] = data_numerical

    return data


def split_data(data, class_column, test_set_size, random_state):
    # бьем данные на входы и выходы
    x = data.drop(class_column, axis=1)
    y = data[class_column]

    # бьем на тестовую и обучающую
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_size, random_state=random_state)

    return {
        'train_set': {
            'x': x_train,
            'y': y_train,
        },
        'test_set': {
            'x': x_test,
            'y': y_test,
        },
    }
