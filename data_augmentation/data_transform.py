import torchvision.transforms as transforms
from .Gaussian_blur import GaussianBlur
#from torch.functional import InterpolationMode


def my_transform1(isOnline):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    if isOnline:
        my_transforms = transforms.Compose([
            #my_transforms,
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=(5, 5)),
            transforms.ToTensor()
        ])
    else:
        my_transforms = transforms.Compose([
            #my_transforms,
            transforms.RandomResizedCrop(size=(224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(5, 5))], p=0.1),
            #transforms.RandomSolarize(threshold=[0, 1], p=0.5),
            transforms.ToTensor()
        ])

    return my_transforms


def my_transform(isOnline):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    my_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=(96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=9),
        transforms.ToTensor()
    ])

    return my_transforms