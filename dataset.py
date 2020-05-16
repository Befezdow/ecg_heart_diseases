import math
import os
import random

import numpy as np
import scipy.io as sio
import torch
import pandas as pd
from torch.utils.data import Dataset


class EcgDataset(Dataset):
    def __init__(self, csv_file, root_dir, excepted_samples=None, transform=None, width=5000, augment_width=2500, augment_multiplier=5):
        excepted_samples_set = set(excepted_samples) if excepted_samples is not None else set()

        self.marks_frame = pd.read_csv(csv_file).fillna(0)
        self.marks_frame = self.marks_frame[~self.marks_frame.Recording.isin(excepted_samples_set)]

        self.root_dir = root_dir
        self.transform = transform

        self.original_dataset_size = len(self.marks_frame)
        self.augment_width = augment_width
        self.augment_multiplier = augment_multiplier
        self.augment_borders = []
        for i in range(augment_multiplier):
            left_border = random.randint(0, width - augment_width)
            right_border = left_border + augment_width
            self.augment_borders.append((left_border, right_border))

    def __len__(self):
        return len(self.marks_frame) * self.augment_multiplier

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        interval_number = math.floor(idx / self.original_dataset_size)
        original_idx = idx - interval_number * self.original_dataset_size

        short_data_filename = self.marks_frame.iloc[original_idx, 0] + '.mat'
        full_data_filename = os.path.join(self.root_dir, short_data_filename)
        raw_data = sio.loadmat(full_data_filename)

        gender = raw_data['ECG'][0][0][0][0]
        gender = 0 if gender == 'Male' else 1   # vectorization

        age = raw_data['ECG'][0][0][1][0][0]
        data = raw_data['ECG'][0][0][2].astype('Float32')

        borders = self.augment_borders[interval_number]
        data = data[:, borders[0]:borders[1]]

        marks = np.array([self.marks_frame.iloc[original_idx, 1:] - 1], dtype="int")
        marks = marks[0][0]    # TODO because right now we are predicting the main diagnosis

        sample = (gender, age, data), marks

        if self.transform:
            sample = self.transform(sample)

        return sample


def detect_small_samples(csv_file, root_dir, req_count):
    marks_frame = pd.read_csv(csv_file).fillna(-1)

    small_samples = []
    for i in range(0, len(marks_frame)):
        mark_name = marks_frame.iloc[i, 0]
        short_data_filename = mark_name + '.mat'
        full_data_filename = os.path.join(root_dir, short_data_filename)
        raw_data = sio.loadmat(full_data_filename)
        data = raw_data['ECG'][0][0][2]

        if data.shape[1] < req_count:
            small_samples.append(mark_name)

    return small_samples


def check_classes_balance(csv_file):
    result = {
        'first_label': {},
        'second_label': {},
        'third_label': {},
    }

    marks_frame = pd.read_csv(csv_file).fillna(-1)
    for i in range(0, len(marks_frame)):
        first_label = int(marks_frame.iloc[i, 1])
        second_label = int(marks_frame.iloc[i, 2])
        third_label = int(marks_frame.iloc[i, 3])

        result['first_label'][first_label] = result['first_label'][first_label] + 1 \
            if result['first_label'].get(first_label) is not None \
            else 1
        result['second_label'][second_label] = result['second_label'][second_label] + 1 \
            if result['second_label'].get(second_label) is not None \
            else 1
        result['third_label'][third_label] = result['third_label'][third_label] + 1 \
            if result['third_label'].get(third_label) is not None \
            else 1

    return result


def check_gender_age_stats(csv_file, root_dir):
    marks_frame = pd.read_csv(csv_file)

    male_count = 0
    female_count = 0
    empty_gender_count = 0
    max_age = 0
    min_age = 1000
    avg_age = 0
    empty_age_count = 0
    negative_age_count = 0

    dataset_size = len(marks_frame)
    for i in range(0, dataset_size):
        mark_name = marks_frame.iloc[i, 0]
        short_data_filename = mark_name + '.mat'
        full_data_filename = os.path.join(root_dir, short_data_filename)
        raw_data = sio.loadmat(full_data_filename)

        gender = raw_data['ECG'][0][0][0][0]
        age = raw_data['ECG'][0][0][1][0][0]

        if gender == 'Male':
            male_count += 1
        elif gender == 'Female':
            female_count += 1
        else:
            empty_gender_count += 1

        if age == np.nan or math.isnan(age):
            empty_age_count += 1
            continue

        if age < 0:
            negative_age_count += 1

        max_age = age if age > max_age else max_age
        min_age = age if age < min_age else min_age
        avg_age += age / dataset_size

    print(f'Males count: {male_count}')
    print(f'Females count: {female_count}')
    print(f'Empty gender count: {empty_gender_count}')
    print(f'Max age: {max_age}')
    print(f'Min age: {min_age}')
    print(f'Average age: {avg_age}')
    print(f'Empty age count: {empty_age_count}')
    print(f'Negative age count: {negative_age_count}')
