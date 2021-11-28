import torch
from torch.functional import Tensor
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import os
import data_augmentation.data_transform as transform
from data_augmentation.multiview_data import MultiviewData
from trainer import BYOLTrainer
from net.resnet_with_projector import my_resnet_with_projector
from net.vgg_with_projector import my_vgg_with_projector
from net.MLP_head import MLPHead
import argparse


torch.manual_seed(0)


def get_parser():
    parser = argparse.ArgumentParser(description='BYOL_parser')
    #DDP
    parser.add_argument('--local_rank',default=-1,type=int,
                        help='node rank for distributed training')
    return parser



def main():
    parser = get_parser()
    opt = parser.parse_args()

    # device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
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

    online_net = my_vgg_with_projector(512, 128)
    online_predictor = MLPHead(online_net.projector.layer[-1].out_features,
                               512, 128)
    target_net = my_vgg_with_projector(512, 128)

    optimizer = torch.optim.Adam(list(online_net.parameters()) +
                                 list(online_predictor.parameters()),
                                 lr,
                                 beta,
                                 weight_decay)
    step_optimizer = StepLR(optimizer, step_size = 10, gamma=0.5)
    momentum_ori = 0.999
    my_trainer = BYOLTrainer(num_epoch, batch_size, online_net,
                             online_predictor, target_net, optimizer, step_optimizer,
                             momentum_ori, checkpoint_path,opt)
    my_trainer.train(data)


if __name__ == '__main__':
    main()