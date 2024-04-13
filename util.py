from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms

import matplotlib.pyplot as plt


def apply_zca(data, zca_mean, zca_components):
        temp = data.numpy()
        #temp = temp.transpose(0,2,3,1)
        shape = temp.shape
        temp = temp.reshape(-1, shape[1]*shape[2]*shape[3])
        temp = np.dot(temp - zca_mean, zca_components.T)
        temp = temp.reshape(-1, shape[1], shape [2], shape[3])
        #temp = temp.transpose(0, 3, 1, 2)
        data = torch.from_numpy(temp).float()
        return data
    
    
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
    
class KDCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform_feature, transform_im):
        self.transform_im = transform_im
        self.transform_feature = transform_feature

    def __call__(self, x):
        hfn = transforms.functional.crop(x, i=0,j=0,h=300,w=1)
        x_a = transforms.functional.crop(x, i=0,j=1,h=300,w=300)
        '''plt.subplot(1,2,1)
        plt.imshow(x_a)
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(x_b)
        plt.show()'''
        return [self.transform_feature(hfn), self.transform_im(x_a)]
    
    
class KDTwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform_feature, transform_A, transform_B):
        self.transform_feature = transform_feature
        self.transform_A = transform_A
        self.transform_B = transform_B

    def __call__(self, x):
        hfn = transforms.functional.crop(x, i=0,j=0,h=300,w=1)
        x_a = transforms.functional.crop(x, i=0,j=1,h=300,w=300)
        x_b = transforms.functional.crop(x, i=0,j=301,h=300,w=300)
        '''plt.subplot(1,2,1)
        plt.imshow(x_a)
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(x_b)
        plt.show()'''
        return [self.transform_feature(hfn), self.transform_A(x_a), self.transform_B(x_b)]


class TwoCropTransform_:
    """Create two crops of the same image"""
    def __init__(self, transform_A, transform_B):
        self.transform_A = transform_A
        self.transform_B = transform_B

    def __call__(self, x):
        x_a = transforms.functional.crop(x, i=0,j=1,h=300,w=300)
        x_b = transforms.functional.crop(x, i=0,j=301,h=300,w=300)
        '''plt.subplot(1,2,1)
        plt.imshow(x_a)
        #plt.show()
        plt.subplot(1,2,2)
        plt.imshow(x_b)
        plt.show()'''
        return [self.transform_A(x_a), self.transform_B(x_b)]


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

        
class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
    
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            #print(f"k: {k}   correct: {correct[:k].size()}")
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def compute_confusion_matrix(pred, truth):
    cm = torch.zeros([6,6])
    with torch.no_grad():
        for i in range(truth.size(0)):
            _, pred_label = pred[i].topk(1)
            truth_label = truth[i]#.topk(1)
            #print(pred_label)
            pred_label = int(pred_label)
            truth_label = int(truth_label)
            cm[truth_label, pred_label] = cm[truth_label, pred_label] + 1
            #print(cm)
    return cm
    
    
def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()
            

def mkdir_p(path):
    '''make dir if not exist'''
    import os
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise