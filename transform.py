import torch
import numpy as np


class CutOff(object):
    def __init__(self, max_width=5000):
        self.max_width = max_width

    def __call__(self, sample):
        (gender, age, height, weight), data, label = sample
        return (gender, age, height, weight), np.delete(data, np.s_[self.max_width:], 1), label


class ToTensor(object):
    def __call__(self, sample):
        (gender, age, height, weight), data, label = sample
        return torch.tensor([gender, age, height, weight]), torch.from_numpy(data), label - 1
