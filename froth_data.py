import random
import numpy as np
import PIL
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision
import torch

from torch.utils.data import Dataset

class TransformOnce:
    """Create one crop of the same image"""
    def __init__(self, transform_A=None, transform_B=None):
        self.transform_A = torchvision.transforms.Compose([
            #transforms.functional.crop(i=0,j=0,h=400,w=400),
            torchvision.transforms.RandomCrop(256),
            #torchvision.transforms.CenterCrop(256),
            #RandAugmentMC(n=1, m=6),
            torchvision.transforms.ToTensor(),
            #normalize,
        ])
        
    def __call__(self, x):
        x_a = torchvision.transforms.functional.crop(x, top=0,left=1,height=300,width=300)
        out = self.transform_A(x_a)
        return out


class TransformTwice:
    """Create two crops of the same image"""
    def __init__(self, transform_A=None, transform_B=None):
        self.transform_A = torchvision.transforms.Compose([
            #transforms.functional.crop(i=0,j=0,h=400,w=400),
            torchvision.transforms.RandomCrop(256),
            #torchvision.transforms.CenterCrop(256),
            torchvision.transforms.ToTensor(),
            #normalize,
        ])
        
        self.transform_B = torchvision.transforms.Compose([
            #transforms.functional.crop(0,400,400,800),
            torchvision.transforms.RandomCrop(256),
            #torchvision.transforms.CenterCrop(256),
            #RandAugmentMC(n=1, m=6),
            torchvision.transforms.ToTensor(),
            #normalize,
        ])

    def __call__(self, x):
        x_a = torchvision.transforms.functional.crop(x, top=0,left=1,height=300,width=300)
        x_b = torchvision.transforms.functional.crop(x, top=0,left=301,height=300,width=300)
        '''plt.subplot(1,2,1)
        plt.imshow(x_a)
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(x_b)
        plt.show()'''
        out1 = self.transform_A(x_a)
        out2 = self.transform_B(x_b)
        return out1, out2

def get_froth_data(transform_train=None, transform_val=None):

    labeled_dataset = torchvision.datasets.ImageFolder(root='/home/neuralits/NeuralITS/code/data/FrothData4SSL/labeled/', transform=TransformOnce())
    unlabeled_dataset = torchvision.datasets.ImageFolder(root='/home/neuralits/NeuralITS/code/data/FrothData4SSL/unlabeled/', transform=TransformTwice())
    test_dataset = torchvision.datasets.ImageFolder(root='/home/neuralits/NeuralITS/code/data/FrothData4SSL/test/', transform=TransformOnce())
    
    #labeled_dataset = torchvision.datasets.ImageFolder(root='/media/neuralits/SmartView/Dataset/FrothData4SSL/labeled/', transform=TransformOnce())
    #unlabeled_dataset = torchvision.datasets.ImageFolder(root='/media/neuralits/SmartView/Dataset/FrothData4SSL/unlabeled/', transform=TransformTwice())
    #test_dataset = torchvision.datasets.ImageFolder(root='/media/neuralits/SmartView/Dataset/FrothData4SSL/test3/', transform=TransformOnce())

    #print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return labeled_dataset, unlabeled_dataset, test_dataset
    

def _float_parameter(v, max_v):
    return float(v) * max_v / 10

def _int_parameter(v, max_v):
    return int(v * max_v / 10)

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)

def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)

def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Equalize(img, **kwarg):  # equalizes the image histogram
    return PIL.ImageOps.equalize(img)

def Identity(img, **kwarg):
    return img

def Posterize(img, v, max_v, bias=0):  # Reduce the number of bits for each color channel
    v = _int_parameter(v, max_v) #+ bias
    return PIL.ImageOps.posterize(img, v)

def Sharpness(img, v, max_v, bias=0):  # 锐度增强
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def fixmatch_augment_pool():
    # FixMatch paper
    augs = [#(AutoContrast, None, None),
            (Brightness, 3.2, 1.6),
            (Color, 3.2, 1.6),
            (Contrast, 2.2, 6.5),
            #(Equalize, None, None),
            (Identity, None, None),
            (Posterize, 6, 6),
            (Sharpness, 0.9, 0.01)]
    return augs

class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 6
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        return img