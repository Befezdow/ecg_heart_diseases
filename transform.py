import torch
import numpy as np


class CutOff(object):
    def __init__(self, max_width=5000):
        self.max_width = max_width

    def __call__(self, sample):
        (gender, age, data), marks = sample
        return (gender, age, np.delete(data, np.s_[self.max_width:], 1)), marks


class ToTensor(object):
    def __call__(self, sample):
        (gender, age, data), marks = sample
        return (torch.tensor([gender, age]), torch.from_numpy(data)), marks
