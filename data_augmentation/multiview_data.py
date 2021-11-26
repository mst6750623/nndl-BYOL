import torch


class MultiviewData():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        out = [transforms(x) for transforms in self.transforms]
        return out