"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
"""
import os
import sys
from torch.utils.data import DataLoader
from timeit import default_timer
from utilities import *
from copy import deepcopy
import h5py
from scipy.io import loadmat
import fourierpack as sp
import functools
import matplotlib

data_name = 'pipe'

from NOs_dict.models import WSWANO, SOL, T_pipe, iT_pipe, Transform, PseudoSpectra, ZerosFilling
from functools import partial as PARTIAL
T = Transform(T_pipe, iT_pipe)

class SOL_drop(nn.Module):
    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL = 0):
        super(SOL_drop, self).__init__()

        modes = np.array([modes]*dim) if isinstance(modes, int) else np.array(modes)
        bandwidth = np.array([bandwidth]*dim) if isinstance(bandwidth, int) else np.array(bandwidth)
        triL = np.array([triL]*dim) if isinstance(triL, int) else np.array(triL)

        self.modes = modes
        self.width = width
        self.triL = triL
        self.T = T
        self.dim = dim
        self.X_dims = np.arange(-dim, 0)
        if dim == 1:
            convND = nn.Conv1d
        elif dim == 2:
            convND = nn.Conv2d
        elif dim == 3:
            convND = nn.Conv3d

        self.conv0 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)
        self.conv1 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)
        self.conv2 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)
        self.conv3 = PseudoSpectra(T, dim, width, width, modes, bandwidth, triL)

        self.convl = PseudoSpectra(T, dim, in_channels, width-in_channels, modes, bandwidth, triL)

        self.w0 = convND(width, width, 1)
        self.w1 = convND(width, width, 1)
        self.w2 = convND(width, width, 1)
        self.w3 = convND(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
        self.skip = nn.Identity() if skip else ZerosFilling()

    def forward(self, x):

        # [batch, XYZ, c] -> [batch, c, XYZ]
        x = x.permute(0, -1, *self.X_dims-1)

        x = torch.cat([x, F.gelu(self.convl(x))], dim=1) # liftting

        x = self.skip(x) + F.gelu(self.w0(x) + self.conv0(x))

        x = self.skip(x) + F.gelu(self.w1(x) + self.conv1(x))

        x = self.skip(x) + F.gelu(self.w2(x) + self.conv2(x))

        x = self.skip(x) + F.gelu(self.w3(x) + self.conv3(x))

        x = x.permute(0, *self.X_dims, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x

NO_DirNeu = PARTIAL(SOL_drop, T, dim=2)

class inhomogeneous_NO(nn.Module): 
## You only need to add one particular solution into the solution space of homogeneous BCs for the inhomogeneous ones
    def __init__(self, homogeneous_NO, particular):
        super().__init__()
        self.homogeneous_NO = homogeneous_NO
        self.particular = particular
    def forward(self, x):
        x = self.homogeneous_NO(x) + self.particular
        return x

#### fixing seeds
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--data-dict', default='/home/father/OPNO/data/pipe/corrected/', type=str, help='dataset folder')
    # parser.add_argument('--data-dict', default='/home/father/OPNO/data/pipe/', type=str, help='dataset folder')
    parser.add_argument('--data-para', default='100.0', type=str, help='dataset parameter beta')
    parser.add_argument('--epochs', default=500, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=4e-3, type=float, help='learning rate')
    parser.add_argument('--bw1', default=3, type=int, help='band width')
    parser.add_argument('--bw2', default=1, type=int, help='band width')
    parser.add_argument('--batch-size', default=20, type=int, help='batch size')
    parser.add_argument('--wd', default=-6, type=float, help='weight decay')
    parser.add_argument('--step-size', default=100, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=24, type=int, help='Fourier-like modes')
    parser.add_argument('--width', default=32, type=int, help='')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--sol-skipflag', default=0, type=int, help='')
    parser.add_argument('--scdl', default='plat', type=str, help='')
    parser.add_argument('--gpu', default='0', type=str, help='')
    parser.add_argument('--suffix', default='-drop', type=str, help='')

    return parser.parse_args()


train_size, test_size = 1000, 1000
args = get_args()
data_dict = args.data_dict
bandwidth1 = args.bw1
bandwidth2 = args.bw2
epochs =  args.epochs  # default 500
step_size = args.step_size  # for StepLR, default 50
batch_size = args.batch_size  # default 100
sub = args.sub  # default 1
learning_rate = args.lr  # default 1e-3
bandwidth1 = args.bw1  # default 3
bandwidth2 = args.bw2  # default 1
modes = args.modes
triL = args.triL
suffix = args.suffix
sol_skipflag = args.sol_skipflag
width = args.width
wd = args.wd
weight_decay = 10 ** wd #1e-4
gamma = 0.5  # for StepLR
scdl = args.scdl

INPUT_X = data_dict+'Pipe_X.npy'
INPUT_Y = data_dict+'Pipe_Y.npy'
OUTPUT_Sigma = data_dict+'Pipe_Q.npy'
num_workers = 0
device = torch.device("cuda:"+args.gpu)


file_name = 'sp-' + data_name + str(sub) + '-bw' + str(bandwidth1) +'_'+ str(bandwidth2)  + '-modes' + str(modes) + '-width' + str(width) \
    +'-triL' + str(triL) + '-wd' + str(wd) + '-' + scdl + suffix
result_PATH = '/home/liuziyuan/OPNO/model/new/' + file_name + '.pkl'

print('data:', data_dict)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs, 'bandwidth', bandwidth1, bandwidth2)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub, 'triL', triL)

import os
if os.path.exists(result_PATH) and epochs > 0:
    print("----------Warning: pre-trained model already exists:")
    print(result_PATH)
    print("----------------------------------------------------")

## main

ntrain, ntest = train_size, test_size

r1 = sub
r2 = sub
s1 = int(((129 - 1) / r1) + 1)
s2 = int(((129 - 1) / r2) + 1)

inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float)
input = torch.stack([inputX, inputY], dim=-1)
data_size = 2000

output = np.load(OUTPUT_Sigma)[:, 0]
output = torch.tensor(output, dtype=torch.float)

x_train = input[:data_size][:ntrain, ::r1, ::r2]
y_train = output[:data_size][:ntrain, ::r1, ::r2]
x_test = input[:data_size][-ntest:, ::r1, ::r2]
y_test = output[:data_size][-ntest:, ::r1, ::r2]
x_train = x_train.reshape(ntrain, s1, s2, -1)
x_test = x_test.reshape(ntest, s1, s2, -1)

_, Nx, Ny, _ = x_train.shape
s = Nx

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False)

hom_model = NO_DirNeu(2, modes, width, bandwidth=[bandwidth1, bandwidth2]).to(device)
comp = y_train[0, 0, :] # a particular solution
comp = comp[..., None].to(device)
model = inhomogeneous_NO(hom_model, comp).to(device)

train_list, loss_list = [], []
if epochs == 0:  # load model
    print('model:' + result_PATH + ' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
print('model parameters number =', count_params(model))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
from Adam import Adam
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        out = model(x).reshape(batch_size, s, s)
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

            out = model(x).reshape(batch_size, s, s)
            test_l2 += myloss(out.reshape(batch_size, -1), y.reshape(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    train_list.append(train_l2)
    loss_list.append(test_l2)

    scheduler.step(train_l2)

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

model.eval()
peer_loss = LpLoss(reduction=False)
test_err = torch.tensor([], device=device)
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(batch_size, s, s)
        # out = y_normalizer.decode(out)
        test_err = torch.cat([test_err,
                              peer_loss(out.reshape(batch_size, -1), y.reshape(batch_size, -1))],
                             dim=0)

print('test_l2 min-max:', test_err.min().item(), test_err.max().item())
print('test_l2:', test_err.mean())
