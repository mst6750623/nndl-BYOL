import torch
import os
import torchvision
import time
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from net.vgg_with_projector import my_vgg_with_projector
from net.resnet_with_projector import my_resnet_with_projector
from net.inference_net import InferenceVGG


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_data = torchvision.datasets.CIFAR10(
        '/mnt/pami23/stma/datasets/cifar10',
        train=True,
        #transform=torchvision.transforms.ToTensor(),
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        download=True)
    test_data = torchvision.datasets.CIFAR10(
        '/mnt/pami23/stma/datasets/cifar10',
        train=False,
        #transform=torchvision.transforms.ToTensor(),
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        download=True)

    online_net_with_projector = models.resnet18(pretrained=False)
    is_train = True
    batch_size = 32
    epoch_num = 10
    if is_train:  # 要训练的话，加载训练好的特征提取层的固定参数
        train_checkpoint_path = os.path.join(
            '/mnt/pami23/stma/checkpoints/myBYOL', 'model.pth')
        checkpoints = torch.load(train_checkpoint_path)
        online_net_with_projector.load_state_dict(
            checkpoints['online_network_state_dict'], strict=False)
        representation_layer = online_net_with_projector
        print(list(representation_layer.children())[-1])
        model = InferenceVGG(representation_layer, 512)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.0001,
                                     weight_decay=0.999)
        model.to(device)
        train(train_data, batch_size, model, epoch_num, optimizer, device)
        print("train complete!")
    else:
        representation_layer = online_net_with_projector
        model = InferenceVGG(representation_layer, 512)
        model.to(device)
        test_checkpoint_path = os.path.join(
            '/mnt/pami23/stma/checkpoints/myBYOL', 'inference.pth')
        if os.path.exist(test_checkpoint_path):
            checkpoints = torch.load(test_checkpoint_path)
            model.load_state_dict(checkpoints)
        else:
            print("inference checkpoints not found!")
            return -1

    inference(test_data, batch_size, model, device)


def train(data, batch_size, model, epoch_num, optimizer, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    data_iter = DataLoader(data, batch_size, shuffle=True, num_workers=4)
    model.train()
    print_freq = 200
    iter = 0
    for epoch in range(epoch_num):
        print("epoch:", epoch)
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            l = cal_loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            prec1 = accuracy(y_hat.data, y)[0]
            losses.update(l.item(), x.size(0))
            top1.update(prec1.item(), x.size(0))
            if iter % print_freq == 0:
                print('loss:val-{losses.val:.3f} avg-{losses.avg:.3f}\t'
                      'top1:val-{top1.val:.3f} avg-{top1.avg:.3f}'.format(
                          losses=losses, top1=top1))
            iter += 1
    checkpoint_path = os.path.join('/mnt/pami23/stma/checkpoints/myBYOL',
                                   'inference_resnet.pth')
    torch.save(model.state_dict(), checkpoint_path)


def inference(data, batch_size, model, device):
    losses = AverageMeter()
    top1 = AverageMeter()

    data_iter = DataLoader(data, batch_size, shuffle=False, num_workers=4)
    model.eval()
    print_freq = 100
    iter = 0
    for x, y in data_iter:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_hat = model(x)
            loss = cal_loss(y_hat, y)
        prec1 = accuracy(y_hat.data, y)[0]
        losses.update(loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        if iter % print_freq == 0:
            print('loss:val-{losses.val:.3f} avg-{losses.avg:.3f}\t'
                  'top1:val-{top1.val:.3f} avg-{top1.avg:.3f}'.format(
                      losses=losses, top1=top1))
        iter += 1
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def cal_loss(y_hat, y):
    return torch.nn.CrossEntropyLoss()(y_hat, y)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    print(output, target, topk, maxk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    time.sleep(10)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()