import os
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from copy import deepcopy
import h5py
from scipy.io import loadmat
import fourierpack as sp
import functools
import matplotlib
from NOs_dict.models import SinNO2d as Model

device = torch.device("cuda")
data_name = 'darcy-bench'

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
    parser.add_argument('--data-para', default='100.0', type=str, help='dataset parameter beta')
    parser.add_argument('--epochs', default=500, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=1, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--wd', default=-4, type=float, help='weight decay')
    parser.add_argument('--step-size', default=100, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=24, type=int, help='Fourier-like modes')
    parser.add_argument('--width', default=32, type=int, help='')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--suffix', default='', type=str, help='')
    parser.add_argument('--sol-skipflag', default=0, type=int, help='')

    return parser.parse_args()

#### parameters settings
args = get_args()

epochs =  args.epochs  # default 500
step_size = args.step_size  # for StepLR, default 50
batch_size = args.batch_size  # default 100
sub = args.sub  # default 1
learning_rate = args.lr  # default 1e-3
bandwidth = args.bw  # default 1
modes = args.modes
triL = args.triL
suffix = args.suffix
sol_skipflag = args.sol_skipflag
width = args.width
wd = -6.0#args.wd
weight_decay = 10 ** wd #1e-4

gamma = 0.5  # for StepLR
train_size, test_size = 9000, 1000
num_workers = 0
suffix = suffix + '-plat'

beta = args.data_para
data_PATH = args.data_dict + '2D_DarcyFlow_beta' + beta + '_Train.hdf5'
file_name = 'sp-' + data_name + str(sub) + '-beta' + beta + '-modes' + str(modes) + '-width' + str(width) \
            + '-bw' + str(bandwidth)+ '-triL' + str(triL) + '-wd' + str(wd) + suffix
result_PATH = 'model/' + file_name + '.pkl'

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
model = Model(3, modes, width, bandwidth, triL=triL, skip=sol_skipflag).to(device)

ntrain, ntest = train_size, test_size
raw_data = h5py.File(data_PATH, 'r')

# pad BC
raw_num, raw_x, raw_y = raw_data['nu'].shape
x_data, y_data = np.zeros([raw_num, raw_x + 2, raw_y + 2], dtype=np.float32), np.zeros(
    [raw_num, raw_x, raw_y], dtype=np.float32)
x_data[..., 1:-1, 1:-1], y_data = raw_data['nu'], raw_data['tensor'][:, 0, ...]

x_train, x_test = torch.from_numpy(x_data[:train_size, ...]), torch.from_numpy(x_data[-test_size:, ...])
y_train, y_test = torch.from_numpy(y_data[:train_size, ...]), torch.from_numpy(y_data[-test_size:, ...])

_, Nx, Ny = x_train.shape
s = Nx

x_train = x_train.reshape(ntrain, s, s, 1)
x_test = x_test.reshape(ntest, s, s, 1)

grid_x = torch.linspace(0, 1, Nx, dtype=torch.float32).view(1, Nx, 1, 1)
grid_y = torch.linspace(0, 1, Ny, dtype=torch.float32).view(1, 1, Ny, 1)
grid_x[:, 1:-1, ...] = torch.tensor(raw_data['x-coordinate']).view(1, Nx-2, 1, 1)
grid_y[..., 1:-1, :] = torch.tensor(raw_data['y-coordinate']).view(1, 1, Ny-2, 1)
x_train = torch.cat([x_train.view(ntrain, Nx, Ny, 1)  # , du_train
                        , grid_x.repeat(ntrain, 1, Ny, 1), grid_y.repeat(ntrain, Nx, 1, 1)], dim=-1).type(
                        torch.float32)
x_test = torch.cat([x_test.view(ntest, Nx, Ny, 1)  # , du_test
                       , grid_x.repeat(ntest, 1, Ny, 1), grid_y.repeat(ntest, Nx, 1, 1)], dim=-1).type(
                        torch.float32)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False)

train_list, loss_list = [], []
if epochs == 0:  # load model
    print('model:' + result_PATH + ' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    print('test_l2:', loader['loss_list'][-1])
    loss_list = loader['loss_list']
print('model parameters number =', count_params(model))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
from Adam import Adam
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-1, patience=50, verbose=True)

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
        out = model(x).reshape(batch_size, s, s)[..., 1:-1, 1:-1]
        mse = F.mse_loss(out.reshape(batch_size, -1), y.view(batch_size, -1), reduction='mean')

        l2 = myloss(out.reshape(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    # scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x).reshape(batch_size, s, s)[..., 1:-1, 1:-1]
            test_l2 += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    scheduler.step(train_l2)

    train_list.append(train_l2)
    loss_list.append(test_l2)

    t2 = default_timer()
    if (ep + 1) % 100 == 0 or ep < 20:
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
              train_mse, train_l2, test_l2)

if epochs >= 500:
    torch.save({
        'model': model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'modes': modes, 'sub': sub,
        'loss_list': loss_list, 'train_list': train_list
    }, result_PATH)

exit()

j = -1
x_unif = torch.linspace(0, 1, Nx)[1:-1]
X, Y = torch.meshgrid(x_unif, x_unif)

peer_loss = LpLoss(reduction=False)
test_err = torch.tensor([], device=device)
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(batch_size, s, s)[..., 1:-1, 1:-1] # remove BC
        test_err = torch.cat([test_err,
                              peer_loss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))],
                             dim=0)
print('test_l2', test_err.sum().item() / test_size)
print('test_l2 min-max:', test_err.min().item(), test_err.max().item())
