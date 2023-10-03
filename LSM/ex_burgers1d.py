"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import sys
sys.path.append("..")
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

from Adam import Adam
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from LSM_1D import *
# from LSM_Irregular_Geo import *


device = torch.device("cuda")
data_name = 'burgers1d'

#### fixing seeds
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################
import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)
    suffix = ''
    # ## sub1
    # parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    # parser.add_argument('--patch-size', default='5', type=str, help='patch size of different dimensions')
    # parser.add_argument('--padding', default='39', type=str, help='padding size of different dimensions')
    # ## sub2
    # parser.add_argument('--sub', default=2, type=int, help='sub-sample on the data')
    # parser.add_argument('--patch-size', default='4', type=str, help='patch size of different dimensions')
    # parser.add_argument('--padding', default='27', type=str, help='padding size of different dimensions')
    # ## sub16
    # parser.add_argument('--sub', default=16, type=int, help='sub-sample on the data')
    # parser.add_argument('--patch-size', default='3', type=str, help='patch size of different dimensions')
    # parser.add_argument('--padding', default='31', type=str, help='padding size of different dimensions')

    ## sub16-2
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--patch-size', default='6', type=str, help='patch size of different dimensions')
    parser.add_argument('--padding', default='31', type=str, help='padding size of different dimensions')
    suffix = '-ps6'

    # ## sub16-3 no-pad
    # parser.add_argument('--sub', default=16, type=int, help='sub-sample on the data')
    # parser.add_argument('--patch-size', default='4', type=str, help='patch size of different dimensions')
    # # parser.add_argument('--padding', default='0', type=str, help='padding size of different dimensions')



    parser.add_argument('--data-dict', default='/home/father/OPNO/data/', type=str, help='dataset folder')
    parser.add_argument('--epochs', default=5000, type=int, help='training iterations')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=500, type=int, help='step size for the StepLR (if used)')

    parser.add_argument('--in_dim', default=1, type=int, help='input data dimension')
    parser.add_argument('--out_dim', default=1, type=int, help='output data dimension')
    parser.add_argument('--d-model', default=32, type=int, help='')

    # parser.add_argument('--num-basis', default=12, type=int, help='number of basis operators')
    parser.add_argument('--num-basis', default=20, type=int, help='number of basis operators')
    suffix = '-modes20' + suffix
    parser.add_argument('--num-token', default=4, type=int, help='number of latent tokens')
    parser.add_argument('--suffix', default=suffix, type=str, help='')
    return parser.parse_args()

args = get_args()
epochs =  args.epochs  # default 3000
step_size = args.step_size  # for StepLR, default 500
batch_size = args.batch_size  # default 20
sub = args.sub  # default 1
learning_rate = args.lr  # default 1e-3
modes = args.num_basis
suffix = args.suffix
width = args.d_model

# epochs = 0

gamma = 0.5
weight_decay = 1e-4
train_size, test_size = 1000, 100

r = sub
h = int(((4097 - 1) / r) + 1)
patch_size = int(args.patch_size)
padding = int(args.padding)
if not ((h+padding)%16 == 0 and (h+padding)//16 % patch_size == 0): # if get rid of right point
    h = 4096 // sub
s = h
Nx = h
# epochs = 0

# epochs=5000

data_PATH = '/home/father/OPNO/data/burgers_neumann.mat'
file_name = 'LSM-' + data_name + str(sub) + '-modes' + str(modes) + '-width' + str(width) +'-Nx' + str(Nx)+ suffix
result_PATH = '/home/father/OPNO/model/new/' + file_name + '.pkl'

print('data:', data_PATH)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub)


import os

if os.path.exists(result_PATH):
    print("----------Warning: pre-trained model already exists:")
    print(result_PATH)

################################################################
# load data and data normalization
################################################################

ntrain, ntest = train_size, test_size

raw_data = h5py.File(data_PATH, 'r')
x, y = raw_data['u0_unif'], raw_data['u1_unif']

train_size, test_size = 1000, 100
ntrain, ntest = train_size, test_size

# sub-sample
x_data = torch.tensor(x[:, ::sub][:, :Nx])#, dtype=torch.float64)
y_data = torch.tensor(y[:, ::sub][:, :Nx])#, dtype=torch.float64)

x_train = x_data[:ntrain, :]
y_train = y_data[:ntrain, :]
x_test = x_data[-ntest:, :]
y_test = y_data[-ntest:, :]

x_train = x_train.reshape(ntrain, s, 1)
x_test = x_test.reshape(ntest, s, 1)

x_train = x_train.reshape(ntrain, Nx, 1).float()
x_test = x_test.reshape(ntest, Nx, 1).float()

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)

################################################################
# training and evaluation
################################################################
model = Model(args).cuda()
print(count_params(model))
print('GPU', count_params(model)* 4 / (1024**2), "MB")

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, div_factor=1e4,
#                                                 pct_start=0.2,
#                                                 final_div_factor=1e4,
#                                                 steps_per_epoch=len(train_loader), epochs=epochs)

myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
train_list, loss_list = [], []

if epochs == 0:
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    train_list, loss_list = loader['train_list'], loader['loss_list']
    print('test_l2:', loss_list[-1])

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    if (ep+1) % 1000 == 0 or ep < 20:
        print(ep, t2 - t1, train_l2, test_l2)
    train_list.append(train_l2)
    loss_list.append(test_l2)


if epochs >= 1000:
    torch.save({
            'model': model.state_dict(), 'loss_list': loss_list, 'train_list': train_list
    }, result_PATH)

model.eval()

peer_loss = LpLoss(reduction=False)
test_err = torch.tensor([], dtype=x_data.dtype).cuda()
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        out = model(x)
        test_err = torch.cat([test_err,
                              peer_loss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))],
                             dim=0)

print('test_l2 min-max:', test_err.min().item(), test_err.max().item())
print('test_l2', test_err.sum()/test_err.shape[0])

xx, y = x_test, y_test
xx = xx.to(device)
with torch.no_grad():
    yy = model(xx).reshape(ntest, -1).cpu()
j = -1

p = yy
p = torch.abs(p[:, :-1] - p[:, 1:])/(2.0/s)
ans, _ = torch.max(p[:, (0, -1)], dim=1)
print('NBC', torch.mean(ans))

halt

# peer_loss = LpLoss(reduction=False)
# test_err = peer_loss(yy.view(y_test[:100, ...].shape[0], -1), y_test[:100, ...].view(y_test[:100, ...].shape[0], -1))
# print(test_err)

j = -1
yy = model(x_test[:100, ...].cuda()).detach().cpu()[..., 0]
x_unif = torch.linspace(0, 1, Nx)
X, Y = torch.meshgrid(x_unif, x_unif)

plt.cla()
j += 1
fig, axs = plt.subplots(2, 2, num=0, clear=True)
im = axs[0][0].contourf(X, Y, y_test[j, ...], cmap=plt.get_cmap('Spectral'))
fig.colorbar(im, ax=axs[0, 0])
im = axs[0][1].contourf(X, Y, yy[j, ...], cmap=plt.get_cmap('Spectral'))
fig.colorbar(im, ax=axs[0, 1])
im = axs[1][0].contourf(X, Y, yy[j, ...] - y_test[j, ...], cmap=plt.get_cmap('Spectral'))
fig.colorbar(im, ax=axs[1, 0])
im = axs[1][1].contourf(X, Y, x_test[j, ..., 0], cmap=plt.get_cmap('Spectral'))
fig.colorbar(im, ax=axs[1, 1])
axs[0][0].set_axis_off()
axs[0][1].set_axis_off()
axs[1][0].set_axis_off()
axs[1][1].set_axis_off()
plt.show()
