"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
"""
import os
import sys
sys.path.append("..")
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer
from utilities import *
from copy import deepcopy
import h5py
from scipy.io import loadmat
import fourierpack as sp
import functools
import matplotlib

from NOs_dict.models import CosNO_II as Model

device = torch.device("cuda")
data_name = 'diff-react'

#### fixing seeds
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

import argparse
def get_args():
    parser = argparse.ArgumentParser('Spectral Operator Learning', add_help=False)

    parser.add_argument('--epochs', default=500, type=int, help='training iterations')
    parser.add_argument('--sub', default=1, type=int, help='sub-sample on the data')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--bw', default=1, type=int, help='band width')
    parser.add_argument('--batch-size', default=5, type=int, help='batch size')
    parser.add_argument('--step-size', default=100, type=int, help='step size for the StepLR (if used)')
    parser.add_argument('--modes', default=24, type=int, help='Fourier-like modes')
    parser.add_argument('--triL', default=0, type=int, help='')
    parser.add_argument('--suffix', default='', type=str, help='')
    parser.add_argument('--scdl', default='plat', type=str, help='')
    parser.add_argument('--sub-t', default=1, type=int, help='')
    parser.add_argument('--init-t', default=10, type=int, help='')
    return parser.parse_args()

class FNODatasetMult(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False, test_ratio=0.1
                 ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        self.file_path = os.path.abspath(saved_folder + filename + ".h5")
        self.t_step = reduced_resolution_t

        # Extract list of seeds
        with h5py.File(self.file_path, 'r') as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        # Open file and read data
        with h5py.File(self.file_path, 'r') as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype='f')
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend(list([0, -1]))
            data = data.permute(permute_idx)

            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(seed_group["grid"]["x"], dtype='f')
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y)
                grid = torch.stack((X, Y), axis=-1)
            elif dim == 3:
                x = np.array(seed_group["grid"]["x"], dtype='f')
                y = np.array(seed_group["grid"]["y"], dtype='f')
                z = np.array(seed_group["grid"]["z"], dtype='f')
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z)
                grid = torch.stack((X, Y, Z), axis=-1)

        return data[..., ::self.t_step, :][..., :self.initial_step, :], data[..., ::self.t_step, :], grid

#### parameters settings
args = get_args()

epochs =  args.epochs  # default 500
step_size = args.step_size  # for StepLR, default 100
batch_size = args.batch_size  # default 5
sub = args.sub  # default 1
learning_rate = args.lr  # default 1e-3
bandwidth = args.bw  # default 1
modes = args.modes
triL = args.triL
scdl = args.scdl
sub_t = args.sub_t
suffix = args.suffix
initial_step = args.init_t

gamma = 0.5  # for StepLR
weight_decay = 1e-4
width = 24
num_workers = 0

data_PATH = args.data_dict + data_name + '.h5'
file_name = 'sp-' + data_name + str(sub) + '-modes' + str(modes) + '-width' + str(width) \
            + '-bw' + str(bandwidth) + '-triL' + str(triL) + '-' + scdl \
            + '-init_t'+ str(initial_step) + '-sub_t' + str(sub_t) + suffix
# file_name = 'sp-diff-react1-modes24-width24-bw1-triL0step'
result_PATH = args.data_dict + 'model/new/' + file_name + '.pkl'

print('data:', data_PATH)
print('result_PATH:', result_PATH)
print('batch_size', batch_size, 'learning_rate', learning_rate, 'epochs', epochs, 'bandwidth', bandwidth)
print('weight_decay', weight_decay, 'width', width, 'modes', modes, 'sub', sub, 'triL', triL)
print('using model: CosNO2d-II')

import os

if os.path.exists(result_PATH):
    print("----------Warning: pre-trained model already exists:")
    print(result_PATH)

## main

# raw_data = h5py.File(data_PATH, 'r')
# raw_data['0000'].keys()

train_data = FNODatasetMult(data_name,
                               saved_folder=args.data_dict,
                               reduced_resolution=sub,
                               reduced_resolution_t=sub_t,
                               reduced_batch=batch_size,
                               initial_step=initial_step)
val_data = FNODatasetMult(data_name,
                             saved_folder=args.data_dict,
                             reduced_resolution=sub,
                             reduced_resolution_t=sub_t,
                             reduced_batch=batch_size,
                             initial_step=initial_step,
                             if_test=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,# num_workers = num_workers,
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,# num_workers = num_workers,
                                         shuffle=False)


train_size, test_size = train_data.data_list.shape[0], val_data.data_list.shape[0]
ntrain, ntest = train_size, test_size
print('size-of-train/val:', train_size, test_size)

training_type = 'autoregressive'
t_train = (101 - 1) // sub_t + 1
myloss = LpLoss(size_average=False)
loss_fn = myloss

model = Model(initial_step*2+2, modes, width, bandwidth, out_channels=2, dim = 2, triL=triL).to(device)

from Adam import Adam
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if scdl == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=1e-1, patience=50, verbose=True)


train_list, loss_list = [], []

if epochs == 0:  # load model
    print('model:' + result_PATH + ' loaded!')
    loader = torch.load(result_PATH)
    model.load_state_dict(loader['model'])
    print('test_l2:', loader['loss_list'][-1])
    # peer_err = loader['test_err']

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
    for xx, yy, grid in train_loader:
        loss = 0

        # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
        # yy: target tensor [b, x1, ..., xd, t, v]
        # grid: meshgrid [b, x1, ..., xd, dims]
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)

        # Initialize the prediction tensor
        pred = yy[..., :initial_step, :]
        # Extract shape of the input tensor for reshaping (i.e. stacking the
        # time and channels dimension together)
        inp_shape = list(xx.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)
        outp_shape = inp_shape[:-1] + [1, -1]

        if training_type in ['autoregressive']:
            # Autoregressive loop
            for t in range(initial_step, t_train):
                # Reshape input tensor into [b, x1, ..., xd, t_init*v]
                inp = xx.reshape(inp_shape)

                # Extract target at current time step
                y = yy[..., t:t + 1, :]

                # Model run
                im = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)

                # Loss calculation
                _batch = im.size(0)
                loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                # Concatenate the prediction at current time step into the
                # prediction tensor
                pred = torch.cat((pred, im), -2)

                # Concatenate the prediction at the current time step to be used
                # as input for the next time step
                xx = torch.cat((xx[..., 1:, :], im), dim=-2)

            train_l2_step += loss.item()
            _batch = yy.size(0)
            _yy = yy[..., initial_step+1:t_train, :]  # if t_train is not -1
            _pred = pred[..., initial_step+1:t_train, :]
            l2_full = loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1))
            train_l2_full += l2_full.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    train_l2 = train_l2_full / ntrain
    if scdl == 'step':
        scheduler.step()
    else:
        scheduler.step(train_l2)

    if True:
        val_l2_step = 0
        val_l2_full = 0
        with torch.no_grad():
            for xx, yy, grid in val_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                grid = grid.to(device)

                if training_type in ['autoregressive']:
                    pred = yy[..., :initial_step, :]
                    inp_shape = list(xx.shape)
                    inp_shape = inp_shape[:-2]
                    inp_shape.append(-1)

                    for t in range(initial_step, yy.shape[-2]):
                        inp = xx.reshape(inp_shape)
                        y = yy[..., t:t + 1, :]
                        # im = model(inp, grid)
                        im = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                        _batch = im.size(0)
                        loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                        pred = torch.cat((pred, im), -2)

                        xx = torch.cat((xx[..., 1:, :], im), dim=-2)

                    val_l2_step += loss.item()
                    _batch = yy.size(0)
                    _pred = pred[..., initial_step:t_train, :]
                    _yy = yy[..., initial_step:t_train, :]
                    val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()

    test_l2 = val_l2_full / ntest
    train_list.append(train_l2)
    loss_list.append(test_l2)

    t2 = default_timer()
    if (ep + 1) % 10 == 0 or ep < 30:
        print(ep, str(t2 - t1)[:4], optimizer.state_dict()['param_groups'][0]['lr'], \
              train_l2, test_l2)


if epochs >= 200:
    torch.save({
        'model': model.state_dict(), 'batch_size': batch_size, 'learning_rate': learning_rate, 'epochs': epochs,
        'weight_decay': weight_decay, 'width': width, 'modes': modes, 'sub': sub,
        'loss_list': loss_list, 'train_list': train_list
    }, result_PATH)



j = -1


j = -1

peer_loss = LpLoss(reduction=False)
test_err = torch.tensor([])
model.eval()
test_loader = torch.utils.data.DataLoader(val_data, batch_size=1,
                                         shuffle=False)
with torch.no_grad():
    for xx, yy, grid in val_loader:
        val_l2_step = 0
        val_l2_full = 0
        inp_shape = list(xx.shape)
        inp_shape = inp_shape[:-2]
        inp_shape.append(-1)
        outp_shape = inp_shape[:-1] + [1, -1]
        loss = 0
        xx, yy, grid = xx.to(device), yy.to(device), grid.to(device)

        if training_type in ['autoregressive']:
            pred = yy[..., :initial_step, :]
            inp_shape = list(xx.shape)
            inp_shape = inp_shape[:-2]
            inp_shape.append(-1)

            for t in range(initial_step, yy.shape[-2]):
                inp = xx.reshape(inp_shape)
                y = yy[..., t:t + 1, :]
                # im = model(inp, grid)
                im = model(torch.cat([inp, grid], dim=-1)).reshape(outp_shape)
                _batch = im.size(0)
                loss += loss_fn(im.reshape(_batch, -1), y.reshape(_batch, -1))

                pred = torch.cat((pred, im), -2)

                xx = torch.cat((xx[..., 1:, :], im), dim=-2)

            val_l2_step += loss.item()
            _batch = yy.size(0)
            _pred = pred[..., initial_step:t_train, :]
            _yy = yy[..., initial_step:t_train, :]
            val_l2_full += loss_fn(_pred.reshape(_batch, -1), _yy.reshape(_batch, -1)).item()
            print(val_l2_full)
            test_err = torch.cat([test_err,
                              torch.tensor([val_l2_full])],
                             dim=0)

print('test_l2', test_err.sum().item() / test_size)
print('test_l2 min-max:', test_err.min().item(), test_err.max().item())



halt

Nx = Ny = 128
nx = np.linspace(-1, 1, Nx)
ny = np.linspace(-1, 1, Ny)
X, Y = np.meshgrid(nx, ny)

# y = y_data[i:i+batch_size, :]
j = -1

j += 1
# fig = plt.figure()
plt.cla()
plt.subplot(2, 4, 1)
plt.pcolor(X, Y, yy[j, ..., initial_step, 0].cpu(), cmap="jet")
plt.colorbar()
plt.subplot(2, 4, 2)
plt.pcolor(X, Y, yy[j, ..., initial_step, 1].cpu(), cmap="jet")
plt.colorbar()
plt.subplot(2, 4, 3)
plt.pcolor(X, Y, yy[j, ..., -1, 0].cpu(), cmap="jet")
plt.colorbar()
plt.subplot(2, 4, 4)
plt.pcolor(X, Y, yy[j, ..., -1, 1].cpu(), cmap="jet")
plt.colorbar()

plt.subplot(2, 4, 5)
plt.pcolor(X, Y, pred[j, ..., initial_step, 0].cpu(), cmap="jet")
plt.colorbar()
plt.subplot(2, 4, 6)
plt.pcolor(X, Y, pred[j, ..., initial_step, 1].cpu(), cmap="jet")
plt.colorbar()
plt.subplot(2, 4, 7)
plt.pcolor(X, Y, pred[j, ..., -1, 0].cpu(), cmap="jet")
plt.colorbar()
plt.subplot(2, 4, 8)
plt.pcolor(X, Y, pred[j, ..., -1, 1].cpu(), cmap="jet")
plt.colorbar()

plt.show()
