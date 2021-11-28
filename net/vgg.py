import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
import torch


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 10, kernel_size=3, padding=1), nn.Softmax())

    def forward(self, x):
        x = self.features(x)
        x = self.reg_layer(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def init_weights(model):  #
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


cfg = {
    'E': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
        512, 'M', 512, 512, 512, 512, 'M'
    ]
}


def VGG_basemodel(pre_train):
    """
    VGG 19-layer model (configuration "E")
    """
    model = VGG(make_layers(cfg['E'], True))
    # model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    if pre_train == 'vgg19':
        print('use pretrained model: ' + pre_train)
        model.load_state_dict(torch.load(
            '/home/lzx/.cache/torch/checkpoints/vgg19-dcbb9e9d.pth'),
                              strict=False)
    elif pre_train == 'kaiming':
        print('use kaiming init')
        init_weights(model)
    else:
        print('use pretrained model: SSL-' + pre_train)
        model.load_state_dict(
            torch.load(pre_train)['online_network_state_dict'], strict=False)

    return model
