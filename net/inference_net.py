import torch
import torch.nn as nn


class InferenceVGG(nn.Module):
    def __init__(self, features, input_dim):
        super(InferenceVGG, self).__init__()
        self.features = features
        #for p in self.features.parameters():
        #    p.requires_grad = False
        self.classification = nn.Sequential(nn.Linear(input_dim, input_dim),
                                            nn.ReLU(),
                                            nn.Linear(input_dim, input_dim),
                                            nn.ReLU(),
                                            nn.Linear(input_dim, 10))

    def forward(self, x):
        temp = self.features(x)
        #print("temp:", temp.shape)
        temp = temp.view(temp.shape[0], -1)
        return self.classification(temp)
