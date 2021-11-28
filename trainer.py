import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR

from net.BYOL import *

writer = SummaryWriter()


class BYOLTrainer:
    def __init__(self, num_epoch, batch_size, online_net, online_predictor,
                 target_net, optimizer, momentum, checkpoint_path, opt):
        #net_param
        self.online_net = online_net
        self.online_predictor = online_predictor
        self.target_net = target_net
        self.momentum = momentum

        #setting
        self.opt = opt

        #train_param
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.step_optimizer = StepLR(self.optimizer, step_size = 15, gamma=0.2)
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def cal_loss(x1, x2):
        assert x1.shape == x2.shape, "error!cannot be calculated!"
        x1_normalize = x1 / torch.norm(x1, dim=1, keepdim=True)
        x2_normalize = x2 / torch.norm(x2, dim=1, keepdim=True)
        #x1_normalize = F.normalize(x1, dim=1)
        #x2_normalize = F.normalize(x2, dim=1)
        return 2 - 2 * (x1_normalize * x2_normalize).sum(dim=-1)

    def update(self, x1_online_output, x2_target_output,x2_online_output,x1_target_output):
        l1 = self.cal_loss(x1_online_output, x2_target_output)
        l1 += self.cal_loss(x2_online_output, x1_target_output)
        return l1.mean()

    def init_BYOL(self, mode = 'init'):
        '''
        mode= 'init' : kaiming_init
        mode = 'resume' : 载入模型
        '''
        net = BYOL_net(self.online_net,self.online_predictor,self.target_net,self.momentum)
        if mode == 'init':
            net.init_target_param()
        return net


    def train(self, dataset):
        
        opt = self.opt
        iter = 0
        l_sum = 0
        tb_log_intv = 10
        #网络初始化
        net = self.init_BYOL(mode='init')
        
        #DDP
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda',opt.local_rank)
        net=net.to(device)
        net = torch.nn.parallel.DistributedDataParallel(net, 
                                    device_ids=[opt.local_rank],
                                    output_device = opt.local_rank)
        #ddp_dataloader 
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_iter = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    sampler = train_sampler,
                                    num_workers=2,
                                    drop_last=False)
        

        for epoch in range(self.num_epoch):
            losses = []
            if opt.local_rank==0:
                print("epoch:", epoch)
            for (x_1, x_2), _ in tqdm(data_iter, desc="Processing:"):
                x1 = x_1.to(device)
                x2 = x_2.to(device)
                x1_on,x2_t,x2_on,x1_t = net(x1,x2)
                #calculate loss
                l = self.update(x1_on,x2_t,x2_on,x1_t)
                losses.append(l.item())
                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                #更新target____ddp需要用module
                net.module.update_target_param()
                

                if opt.local_rank == 0:
                    if iter !=0 and iter % tb_log_intv == 0:
                        avgl = np.mean(losses[-tb_log_intv:])
                        print('loss:{}'.format(avgl))
                        writer.add_scalar("iter_Loss", avgl, global_step = iter)
                    iter += 1
            if opt.local_rank==0:
                print('total_loss:{}'.format(np.mean(losses)))
                writer.add_scalar("epoch_Loss", np.mean(losses), global_step = epoch)
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                writer.add_scalar('lr',current_lr,global_step=epoch)
                if epoch!=1 and epoch%5:
                    net.module.save_model(os.path.join(self.checkpoint_path, 'biglr001_'+str(epoch)+'model.pth'))
            self.step_optimizer.step()
        if opt.local_rank==0:
            writer.flush()
            writer.close()
