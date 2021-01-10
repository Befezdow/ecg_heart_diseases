import torch
from torch.utils import data
from torchvision import transforms

from dataset import EcgDataset, detect_small_samples
from transform import CutOff, ToTensor


class DataManager:
    def __init__(self, marks_csv='data/REFERENCE.csv', train_dir='data/samples', batch_size=32, sample_width=5000, augment_width=2500, augment_multiplier=5):
        self._train_dir = train_dir
        self._marks_csv = marks_csv
        self._batch_size = batch_size
        self._sample_width = sample_width
        self._augment_width = augment_width
        self._augment_multiplier = augment_multiplier

        self._cnn_transforms = transforms.Compose([
            CutOff(sample_width), ToTensor()
        ])

        self.cuda_params = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

    def get_cnn_train_test_loaders(self, test_part=0.2, need_shuffle=True):
        excepted_samples = detect_small_samples(self._marks_csv, self._train_dir, self._sample_width)
        dataset = EcgDataset(
            self._marks_csv,
            self._train_dir,
            excepted_samples=excepted_samples,
            transform=self._cnn_transforms,
            width=self._sample_width,
            augment_width=self._augment_width,
            augment_multiplier=self._augment_multiplier
        )

        dataset_size = len(dataset)
        test_size = int(dataset_size * test_part)
        train_size = dataset_size - test_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=need_shuffle, **self.cuda_params
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self._batch_size, shuffle=need_shuffle, **self.cuda_params
        )

        return train_loader, test_loader
