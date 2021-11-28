import torch
import torch.nn as nn


class InferenceVGG(nn.Module):
    def __init__(self, features):
        super(InferenceVGG, self).__init__()
        self.features = features
        for p in self.features.parameters():
            p.requires_grad = False
        self.classification = nn.Sequential(
            nn.Conv2d(512, 10, kernel_size=3, padding=1), nn.Softmax())

    def forward(self, x):
        temp = self.features(x)
        return self.classification(temp)
