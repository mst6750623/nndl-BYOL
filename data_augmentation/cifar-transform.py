import torchvision.transforms as transforms
from data_augmentation.Gaussian_blur import GaussianBlur
#from torch.functional import InterpolationMode


def my_transform(isOnline):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    my_transforms = transforms.Compose([
        #$transforms.RandomResizedCrop(size=(96, 96)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=9),
        transforms.ToTensor()
    ])

    return my_transforms