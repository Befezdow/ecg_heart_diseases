import torch
from torch.utils import data
from torchvision import transforms

from dataset import EcgDataset
from transform import ToTensor


class DataManager:
    def __init__(self, train_dir='data/train', test_dir='data/test', labels_file='labels.csv', data_folder='samples', batch_size=32):
        self._train_dir = train_dir
        self._test_dir = test_dir
        self._labels_file = labels_file
        self._data_folder = data_folder
        self._batch_size = batch_size

        self._cnn_transforms = transforms.Compose([ToTensor()])

        self.cuda_params = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

    def get_train_loader(self, need_shuffle=True):
        dataset = EcgDataset(
            root_dir=self._train_dir,
            labels_file=self._labels_file,
            data_folder=self._data_folder,
            transform=self._cnn_transforms,
        )

        return torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=need_shuffle, **self.cuda_params
        )

    def get_test_loader(self, need_shuffle=True):
        dataset = EcgDataset(
            root_dir=self._test_dir,
            labels_file=self._labels_file,
            data_folder=self._data_folder,
            transform=self._cnn_transforms,
        )

        return torch.utils.data.DataLoader(
            dataset, batch_size=self._batch_size, shuffle=need_shuffle, **self.cuda_params
        )