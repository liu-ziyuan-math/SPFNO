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
from NOs_dict.models import CosNO1d as Model

device = torch.device("cuda")
data_name = 'burgers_neumann'

#### fixing seeds
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

from utilities import get_args
import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='data/', type=str, help='dataset folder')
    parser.add_argument('--data-name', default='burgers_neumann.m', type=str, help='dataset name')
    parser.add_argument('--epochs', default=5000, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=4, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--step-size', default=500, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=20, type=int, help='Fourier-like modes')
    parser.add_argument('--suffix', default='', type=str, help='')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--scdl', default='step', type=str, help='')

    return parser.parse_args()

## parameters
args = get_args()

epochs =  args.epochs  # default 5000
step_size = args.step_size  # for StepLR, default 500
batch_size = args.batch_size  # default 20
sub = args.sub  # default 1
learning_rate = args.lr  # default 1e-3
bandwidth = args.bw  # default 1
modes = args.modes
suffix = args.suffix
triL = args.triL
scdl = args.scdl

gamma = 0.5  # for StepLR
weight_decay = 1e-4
train_size, test_size = 1000, 100
width = 50
num_workers = 0

data_PATH = args.data_dict + data_name + '.mat'
file_name = 'sp-' + data_name + str(sub)  + '-modes' + str(modes)  + '-width' + str(width) + '-bw' + str(bandwidth) + '-triL' + str(triL)  + '-' + scdl + suffix
result_PATH = 'model/' + file_name + '.pkl'

if os.path.exists(result_PATH):
    print("-"*40+"\nWarning: pre-trained model already exists:\n"+result_PATH+"\n"+"-"*40)

print('data:', data_PATH)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs, 'bandwidth', bandwidth)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub, 'triL', triL)

raw_data = h5py.File(data_PATH, 'r')
x_data, y_data = raw_data['u0_unif'], raw_data['u1_unif']
x_data, y_data = torch.tensor(x_data[:, ::sub]), torch.tensor(y_data[:, ::sub])

data_size, Nx = x_data.shape
print('data size = ', data_size, 'training size = ', train_size, 'test size = ', test_size, 'Nx = ', Nx)

grid = torch.linspace(-1, 1, Nx, dtype=torch.float64).reshape(1, Nx, 1)
x_data = torch.cat([x_data.reshape(data_size, Nx, 1), grid.repeat(data_size, 1, 1)], dim=2)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[:train_size, :, :], y_data[:train_size, :]), num_workers = num_workers,
        batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[-test_size:, :, :], y_data[-test_size:, :]), num_workers = num_workers,
        batch_size=batch_size, shuffle=False)

## model

model = Model(2, modes, width, bandwidth, triL=triL).to(device).double()

if epochs == 0:  # load model
    print('pretrained model:' + result_PATH + ' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    loss_list = loader['loss_list']
print('model parameters number =', count_params(model))

## training
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
from Adam import Adam

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if scdl == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-1, patience=30, verbose=True)

train_list, loss_list = [], []
t1 = default_timer()

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse, train_l2 = 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        # mse.backward()
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    train_mse /= len(train_loader)
    train_l2 /= train_size
    train_list.append(train_l2)

    if scdl == 'step':
        scheduler.step()
    else:
        scheduler.step(train_l2)

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    test_l2 /= test_size

    loss_list.append(test_l2)

    t2 = default_timer()
    if (ep + 1) % 1000 == 0 or (ep < 30):
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'],
              train_mse, train_l2, test_l2)

## save results
import inspect

current_code = inspect.getsource(inspect.currentframe())
if epochs >= 5000:
    torch.save({
        'model': model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'modes': modes, 'sub': sub,
        'loss_list': loss_list, 'train_list': train_list, 'code': current_code
    }, result_PATH)

## result visualization
xx, y = x_data[-test_size:, ...].to(device), y_data[-test_size:, :]
with torch.no_grad():
    yy = model(xx).reshape(test_size, -1).cpu()

## Neumann loss
p = sp.fourier_partial(yy, sp.cos_transform, sp.isin_transform)
p = p[:, (0, -1)]
ans, _ = torch.max(torch.abs(p), dim=1)
print('BC error:', torch.mean(ans))

peer_loss = LpLoss(reduction=False)
test_err = peer_loss(yy.view(y.shape[0], -1), y.view(y.shape[0], -1))
print('l2 error v.s. max error', str(test_err.sum().item()/test_size)[:20], test_err.max().item())

colors = [' ', "r", 'g', 'b', 'purple']
show = my_plt1d(grid.reshape(-1), myloss, 'SPFNO')
j = -1

######## copy the following code and manually plot the j-th instance
plt.figure(figsize=(14, 10))
j += 1
x, y = x_data[-test_size+j:-test_size+j+1, ...], y_data[-test_size+j, ...]
show.ppt(model, x.to(device), y)
# plt.show()
savefig('spfno')
