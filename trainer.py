import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class BYOLTrainer:
    def __init__(self, num_epoch, batch_size, online_net, online_predictor,
                 target_net, optimizer, momentum, device):
        self.num_epoch = num_epoch
        self.online_net = online_net
        self.online_predictor = online_predictor
        self.target_net = target_net
        self.optimizer = optimizer
        self.momentum = momentum
        self.device = device
        self.batch_size = batch_size

    def init_target_param(self):
        for param_theta, param_fi in zip(self.online_net.parameters(),
                                         self.target_net.parameters()):
            param_fi.data.copy_(param_theta.data)
            param_fi.requires_grad = False

    @torch.no_grad()
    def update_target_param(self):
        for param_theta, param_fi in zip(self.online_net.parameters(),
                                         self.target_net.parameters()):
            param_fi.data = self.momentum * param_fi.data + (
                1. - self.momentum) * param_theta.data

    def save_model(self, PATH):
        torch.save(
            {
                'online_network_state_dict': self.online_net.state_dict(),
                'target_network_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)

    @staticmethod
    def cal_loss(x1, x2):
        assert x1.shape == x2.shape, "error!cannot be calculated!"
        x1_normalize = x1 / torch.norm(x1, dim=1, keepdim=True)
        x2_normalize = x2 / torch.norm(x2, dim=1, keepdim=True)
        #x1_normalize = F.normalize(x1, dim=1)
        #x2_normalize = F.normalize(x2, dim=1)
        return 2 - 2 * (x1_normalize * x2_normalize).sum(dim=-1)

    def update(self, x1, x2):
        x1_online_output = self.online_predictor(self.online_net(x1))
        x2_online_output = self.online_predictor(self.online_net(x2))
        with torch.no_grad():
            x1_target_output = self.target_net(x1)
            x2_target_output = self.target_net(x2)
        l1 = self.cal_loss(x1_online_output, x2_target_output)
        l1 += self.cal_loss(x2_online_output, x1_target_output)
        return l1.mean()

    def train(self, dataset):
        self.data_iter = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    drop_last=False)
        self.init_target_param()
        iter = 0
        l_sum = 0
        for i in range(self.num_epoch):
            print("epoch:", i)
            for (x_1, x_2), _ in tqdm(self.data_iter, desc="Processing:"):
                #print(x[0].shape, x[1].shape, x[0].equal(x[1]))
                x1 = x_1.to(self.device)
                x2 = x_2.to(self.device)
                l = self.update(x1, x2)
                writer.add_scalar('loss', l, global_step=iter)
                #writer.add_scalar("Loss/train", l, i)

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                l_sum += l
                if iter % 100 == 0:
                    print(l_sum / 100)
                    #writer.add_scalar("Loss/train", l_sum / 100, iter)
                    l_sum = 0

                self.update_target_param()

                iter += 1
            #print(l_sum.mean())
        writer.flush()
        self.save_model(
            os.path.join('/mnt/pami23/stma/checkpoints/myBYOL', 'model.pth'))
        writer.close()
