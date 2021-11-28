import torch
from torch import nn

class BYOL_net(nn.Module):
    def __init__(self,online_net,online_predictor,target_net,momentum=0.999):
        super(BYOL_net, self).__init__()
        self.online_net = online_net
        self.online_predictor = online_predictor
        self.target_net = target_net
        self.momentum =momentum
    
    def save_model(self, PATH):
        torch.save(
            {
                'online_network_state_dict': self.online_net.state_dict(),
                'target_network_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, PATH)

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

    def forward(self,x1,x2):
        #online network forward
        x1_online_output = self.online_predictor(self.online_net(x1))
        x2_online_output = self.online_predictor(self.online_net(x2))
        with torch.no_grad():
            x1_target_output = self.target_net(x1)
            x2_target_output = self.target_net(x2)
        return x1_online_output,x2_target_output,x2_online_output,x1_target_output