"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
"""
import os
import sys
sys.path.append("..")
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from copy import deepcopy
import h5py
from scipy.io import loadmat
import fourierpack as sp
import functools

import matplotlib

# device = torch.device("cuda:0")
device = torch.device("cuda")
data_name = 'burgers2d'

#### fixing seeds
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

from utilities import get_args
import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='/home/father/OPNO/data/', type=str, help='dataset folder')
    parser.add_argument('--epochs', default=3000, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=4, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=500, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=16, type=int, help='Fourier-like modes')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--suffix', default='', type=str, help='')
    parser.add_argument('--scdl', default='step', type=str, help='')
    return parser.parse_args()

#### parameters settings
args = get_args()

epochs =  args.epochs  # default 500
step_size = args.step_size  # for StepLR, default 50
batch_size = args.batch_size  # default 20
sub = args.sub  # default 1
learning_rate = args.lr  # default 1e-3
bandwidth = args.bw  # default 1
modes = args.modes
triL = args.triL
suffix = args.suffix
scdl = args.scdl

gamma = 0.5  # for StepLR
weight_decay = 1e-4
train_size, test_size = 1000, 100
width = 24
num_workers = 1

#pycharm
if sys.argv[0][:5] == '/home':
    print('------PYCHARM test--------')
    bandwidth = 5
    sub = 4
    epochs = 0

    bandwidth = 4
    sub = 4
    epochs = 500
    scdl = 'plat'

data_PATH = args.data_dict + data_name + '.mat'
file_name = 'sp-' + data_name + str(sub) + '-modes' + str(modes) + '-width' + str(width) + \
            '-bw' + str(bandwidth) + '-triL' + str(triL) # + '-stepsize'+str(step_size)+suffix
file_name += '-' + scdl
result_PATH = '/home/father/OPNO/model/new/' + file_name + '.pkl'

print('data:', data_PATH)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs, 'bandwidth', bandwidth)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub, 'triL', triL)

import os

if os.path.exists(result_PATH):
    print("----------Warning: pre-trained model already exists:")
    print(result_PATH)

## main

## model
from NOs_dict.models import CosNO2d
model = CosNO2d(3, modes, width, bandwidth, out_channels=3, triL=triL).double().to(device)

ntrain, ntest = train_size, test_size
raw_data = h5py.File(data_PATH, 'r')

x_data, y_data = np.array(raw_data['u_unif'], dtype=np.float64), np.array(raw_data['u_unif'], dtype=np.float64)
x_data = x_data[..., 0]
y_data = y_data[..., (2, 6, 10)]
x_data, y_data = torch.from_numpy(x_data),  torch.from_numpy(y_data)

x_train, x_test = x_data[:ntrain,::sub,::sub, ...], x_data[-ntest:,::sub,::sub, ...]
y_train, y_test = y_data[:ntrain,::sub,::sub, ...], y_data[-ntest:,::sub,::sub, ...]


_, Nx, Ny = x_train.shape

x_train = x_train.reshape(ntrain, Nx, Ny, 1)
x_test = x_test.reshape(ntest, Nx, Ny, 1)

grid_x = torch.linspace(0, 1, Nx, dtype=torch.float64).view(1, Nx, 1, 1)
grid_y = torch.linspace(0, 1, Ny, dtype=torch.float64).view(1, 1, Ny, 1)
x_train = torch.cat([x_train.view(ntrain, Nx, Ny, 1)  # , du_train
                        , grid_x.repeat(ntrain, 1, Ny, 1), grid_y.repeat(ntrain, Nx, 1, 1)], dim=-1).type(
                        torch.float64)
x_test = torch.cat([x_test.view(ntest, Nx, Ny, 1)  # , du_test
                       , grid_x.repeat(ntest, 1, Ny, 1), grid_y.repeat(ntest, Nx, 1, 1)], dim=-1).type(
                        torch.float64)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False)

if epochs == 0:  # load model
    print('model:' + result_PATH + ' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    print('test_l2:', loader['loss_list'][-1])
    # peer_err = loader['test_err']
print('model parameters number =', count_params(model))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
from Adam import Adam
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if scdl == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-1, patience=50, verbose=True)

train_list, loss_list = [], []
t1 = default_timer()

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, Nx, Ny, 3)
        mse = F.mse_loss(out.reshape(batch_size, -1), y.view(batch_size, -1), reduction='mean')

        l2 = myloss(out.reshape(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).reshape(batch_size, Nx, Ny, 3)
            test_l2 += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    train_list.append(train_l2)
    loss_list.append(test_l2)

    if scdl == 'step':
        scheduler.step()
    else:
        scheduler.step(train_l2)


    t2 = default_timer()
    if (ep + 1) % 100 == 0 or ep < 20:
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
              train_mse, train_l2, test_l2)

## save results
import inspect

current_code = inspect.getsource(inspect.currentframe())
if epochs >= 1000:
    torch.save({
        'model': model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'modes': modes, 'sub': sub,
        'loss_list': loss_list, 'train_list': train_list, 'code': current_code
    }, result_PATH)

j = -1
yy = model(x_test[:100, ...].to(device)).detach().cpu()[..., 0]
x_unif = torch.linspace(0, 1, Nx)[1:-1]
X, Y = torch.meshgrid(x_unif, x_unif)

peer_loss = LpLoss(reduction=False)
test_err = torch.tensor([], device=device, dtype=yy.dtype)
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(batch_size, Nx, Ny, 3)
        test_err = torch.cat([test_err,
                              peer_loss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))],
                             dim=0)
print('test_l2', test_err.sum().item() / test_size)
print('test_l2 min-max:', test_err.min().item(), test_err.max().item())

exit()

nx = np.linspace(-1, 1, Nx)
ny = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(nx, ny)

i = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=ntest,
                                          shuffle=False)
# xx = x_train[i:i+batch_size, :, :].to(device)
xx, y = next(iter(test_loader))
xx = xx.to(device)
yy = model(xx).reshape(ntest, Nx, Ny, -1)

# yy = y_normalizer.decode(yy)
yy = yy.detach().to('cpu')

# y = y_data[i:i+batch_size, :]
j = -1

j += 1
# fig = plt.figure()
plt.cla()
for t in range(3):
    plt.subplot(3, 3, 1+t)
    plt.pcolor(X, Y, yy[j, ..., t].cpu(), cmap="jet")
    plt.colorbar()
    plt.subplot(3, 3, 4+t)
    plt.pcolor(X, Y, yy[j, ..., t].cpu(), cmap="jet")
    plt.colorbar()
    plt.subplot(3, 3, 7+t)
    plt.pcolor(X, Y, (y[j, ..., t]-yy[j, ..., t]).cpu(), cmap="jet")
    plt.colorbar()
plt.show()


# plt.subplot(3, 3, 1)
# plt.pcolor(X, Y, yy[j, ..., -1].cpu(), cmap="jet")
# plt.colorbar()
# plt.subplot(3, 3, 2)
# plt.pcolor(X, Y, yy[j, ..., -1].cpu(), cmap="jet")
# plt.colorbar()
#
# plt.subplot(3, 2, 3)
# plt.pcolor(X, Y, yy[j, ..., 1].cpu(), cmap="jet")
# plt.colorbar()
# plt.subplot(3, 2, 4)
# plt.pcolor(X, Y, yy[j, ..., 1].cpu(), cmap="jet")
# plt.colorbar()
# plt.subplot(3, 2, 5)
# plt.pcolor(X, Y, y[j, ..., 2].cpu(), cmap="jet")
# plt.colorbar()
# plt.subplot(3, 2, 6)
# plt.pcolor(X, Y, y[j, ..., 2].cpu(), cmap="jet")
# plt.colorbar()
# plt.show()