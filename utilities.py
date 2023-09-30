import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
import matplotlib.pyplot as plt
import operator

import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='/home/father/OPNO/data/', type=str, help='dataset folder')
    parser.add_argument('--data-name', default='burgers_neumann.m', type=str, help='dataset name')
    parser.add_argument('--epochs', default=5000, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=1, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=500, type=int, help='step size for the StepLR (if used)')

    return parser.parse_args()

pic_path = '/home/father/Nutstore Files/code/'

def savefig(name='x'):
    if type(name) == type(1):
        plt.savefig('/home/father/temp.png', bbox_inches='tight', pad_inches=0.2, dpi=600)
    else:
        plt.savefig(pic_path + name, bbox_inches='tight', pad_inches=0.2, dpi=600)

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)

        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class my_plt1d(object):
    def __init__(self, mesh, loss_fun, label='OPNO', clr='r'):
        super().__init__()

        self.label = label
        self.mesh = mesh

        with torch.no_grad():
            self.lossfun = loss_fun
            self.clr = clr

    def ppt(self, model, x, y):
        plt.cla()
        with torch.no_grad():
            yy = model(x).reshape(-1).cpu()
        plt.scatter(self.mesh, yy, color=self.clr, s=200, alpha=0.75, label=self.label)
        plt.plot(self.mesh, y, color='b', label='$u_1$ ref',
                    linewidth=2)
        plt.plot(self.mesh, x[0, ..., 0].cpu(), ':', label='$u_0$', linewidth=5)
        print(self.lossfun(yy.reshape(1, -1), y.reshape(1, -1)))

        plt.tick_params(labelsize=40)
        plt.legend(fontsize=30)

