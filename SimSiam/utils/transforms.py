import torchvision.transforms as transforms
from torch.nn import Unfold
from PIL import ImageFilter
import random
import torch.nn as nn

#parts from https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py

class GaussianBlur(object):
    """Gaussian blur augmentation SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TransformsSimSiam(object):

    # from SimSiam https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
    def __init__(self, args):


        #CIFAR : (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        #IN1K : mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #normalisation IN1K
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.augmentation = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize
        ])
    
    def __call__(self, x):

        q = self.augmentation(x)
        k = self.augmentation(x)

        return [q, k]


class TransformsCifar(object):

    def __init__(self, args):


        #transforms.Resize((args.img_size, args.img_size), interpolation=3, antialias=False),

        self.transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
    
    def __call__(self, img):
        return self.transforms(img)

