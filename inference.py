import torch
import os
import torchvision
from torch.utils.data import DataLoader
from net.vgg_with_projector import my_vgg_with_projector
from net.inference_net import InferenceVGG


def inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = torchvision.datasets.CIFAR10(
        'mnt/pami23/stma/datasets/cifar10',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    test_data = torchvision.datasets.CIFAR10(
        'mnt/pami23/stma/datasets/cifar10',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    online_net_with_projector = my_vgg_with_projector(512, 128).to(device)
    checkpoint_path = os.path.join('/mnt/pami23/stma/checkpoints/myBYOL',
                                   'model.pth')
    '''checkpoints = torch.load(checkpoint_path)

    online_net_with_projector.load_state_dict(
        checkpoints['online_network_state_dict'])'''
    representation_layer = online_net_with_projector.encoder
    model = InferenceVGG(representation_layer)
    batch_size = 32
    epoch_num = 10
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.0001,
                                 weight_decay=0.999)
    train(train_data, batch_size, model, epoch_num, optimizer)


def train(data, batch_size, model, epoch_num, optimizer):
    data_iter = DataLoader(data, batch_size, shuffle=True, num_workers=4)
    for epoch in range(epoch_num):
        for x, y in data_iter:
            y_hat = model(x)
            l = cal_loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


def cal_loss(y_hat, y):
    return torch.nn.CrossEntropyLoss(y_hat, y)


if __name__ == '__main__':
    inference()