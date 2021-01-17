import os
import torch
import pandas as pd
from torch.utils.data import Dataset


class EcgDataset(Dataset):
    def __init__(self, root_dir, labels_file, data_folder, transform=None):
        labels_file_path = os.path.join(root_dir, labels_file)
        self.common_frame = pd.read_csv(labels_file_path, header=None)

        self.data_dir = os.path.join(root_dir, data_folder)
        self.transform = transform

    def __len__(self):
        return self.common_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        needed_row = self.common_frame.iloc[idx]

        data_file = str(needed_row[0].astype(int)) + '.csv'
        label = needed_row[1]
        gender = needed_row[2]
        age = needed_row[3]
        height = needed_row[4]
        weight = needed_row[5]

        data_file_path = os.path.join(self.data_dir, data_file)
        data_frame = pd.read_csv(data_file_path, header=None)

        sample = (gender, age, height, weight), data_frame.to_numpy(dtype='float32').transpose(), label

        if self.transform:
            sample = self.transform(sample)

        return sample
