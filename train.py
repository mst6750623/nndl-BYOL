import torch
from torch.functional import Tensor
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import os
import data_augmentation.data_transform as transform
from data_augmentation.multiview_data import MultiviewData
from trainer import BYOLTrainer
from net.resnet_with_projector import my_resnet_with_projector
from net.vgg_with_projector import my_vgg_with_projector
from net.MLP_head import MLPHead

torch.manual_seed(0)


def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    data_root = "/mnt/pami23/longzili/DATA/nndl_BYOL"
    checkpoint_path = '/mnt/pami23/longzili/checkfolder/nndl_BYOL/'
    num_epoch = 40
    batch_size = 256

    #optimizer
    lr=0.0001
    beta=(0.5,0.999)
    weight_decay=0.99

    data = torchvision.datasets.STL10(
        data_root,
        split='train+unlabeled',
        transform=MultiviewData(
            [transform.my_transform(True),
             transform.my_transform(False)]),
        #transform=torchvision.transforms.ToTensor(),
        download=True)

    online_net = my_vgg_with_projector(512, 128).to(device)
    online_predictor = MLPHead(online_net.projector.layer[-1].out_features,
                               512, 128).to(device)
    target_net = my_vgg_with_projector(512, 128).to(device)

    optimizer = torch.optim.Adam(list(online_net.parameters()) +
                                 list(online_predictor.parameters()),
                                 lr,
                                 beta,
                                 weight_decay)
    momentum_ori = 0.999
    my_trainer = BYOLTrainer(num_epoch, batch_size, online_net,
                             online_predictor, target_net, optimizer,
                             momentum_ori, device, checkpoint_path)
    my_trainer.train(data)


if __name__ == '__main__':
    main()