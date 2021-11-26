import torch
import torch.nn as nn
import torchvision.models as models
import net.vgg as vgg
from net.MLP_head import MLPHead


class my_resnet_with_projector(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(my_resnet_with_projector, self).__init__()
        #self.resnet = models.resnet18(pretrained=False)
        #self.encoder = nn.Sequential(*list(self.resnet.children())[:-1])
        self.encoder = vgg.basemodel('kaiming').features
        self.projector = MLPHead(512, hidden_channels, out_channels)

    def forward(self, x):
        temp = self.encoder(x)
        #print(temp.shape)
        temp = temp.view(temp.shape[0], temp.shape[1])
        #print(temp.shape)
        return self.projector(temp)
