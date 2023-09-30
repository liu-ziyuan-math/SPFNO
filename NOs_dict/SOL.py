"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn

We aim to offer a universal platform for all SOL-like NOs but it fails for now because
currently only 4-D input tensors (batched image-like tensors) are supported for native nn.unfold

"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import functools

class Transform:
    def __init__(self, fwd, inv):
        assert (type(fwd)== functools.partial and type(inv) == functools.partial)
        self.fwd = fwd
        self.inv = inv
    def __call__(self, *args, **kwargs):
        return self.fwd(*args, **kwargs)


class PseudoSpectra(nn.Module):
    def __init__(self, T, dim, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super(PseudoSpectra, self).__init__()

        self.T = T
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-dim, 0)

        # print([(l, 0) for l in triL])
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels*bandwidth.prod().item(), out_channels, modes.prod().item()))
        self.unfold = torch.nn.Unfold(kernel_size=bandwidth,
                                      padding=triL)
        self.X_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes]
        self.pad_slices = [slice(None), slice(None)] + [slice(freq) for freq in modes+bandwidth-1-triL*2]

    def quasi_diag_mul(self, input, weights):
        xpad = self.unfold(input)
        return torch.einsum("bix, iox->box", xpad, weights)

    def forward(self, u):
        batch_size = u.shape[0]

        b = self.T(u, self.X_dims)

        out = torch.zeros(batch_size, self.out_channels, *u.shape[2:], device=u.device, dtype=u.dtype)
        out[self.X_slices] = self.quasi_diag_mul(b[self.pad_slices], self.weights).reshape(
                batch_size, self.out_channels, *self.modes)

        u = self.T.inv(out, self.X_dims)
        return u


class ZerosFilling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.zeros(1, device = x.device)

class SOL(nn.Module):
    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL = 0):
        super(SOL, self).__init__()

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

        x = torch.cat([x, F.gelu(self.convl(x))], dim=1)

        x = self.skip(x) + F.gelu(self.w0(x) + self.conv0(x))

        x = self.skip(x) + F.gelu(self.w1(x) + self.conv1(x))

        x = self.skip(x) + F.gelu(self.w2(x) + self.conv2(x))

        x = self.skip(x) + F.gelu(self.w3(x) + self.conv3(x))

        x = x.permute(0, *self.X_dims, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = self.T(self.T.inv(x, self.X_dims-1), self.X_dims-1)

        return x


# temporary code for 3D-unfold
class PseudoSpectra1d(nn.Module):
    def __init__(self, T, in_channels, out_channels, modes, bandwidth=1, triL=0):
        super().__init__()

        self.T = T
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.bandwidth = bandwidth
        self.triL = triL
        self.X_dims = np.arange(-1, 0)

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(modes, in_channels, out_channels, bandwidth))
        # self.unfold = torch.nn.Unfold(kernel_size=bandwidth,
        #                               padding=triL)

    def quasi_diag_mul(self, x, weights):
        xpad = x.unfold(-1, self.bandwidth, 1)
        return torch.einsum("bixw, xiow->box", xpad, weights)

    def forward(self, u):
        batch_size, _, Nx= u.shape

        b = self.T(u, self.X_dims)

        out = torch.zeros((batch_size, self.out_channels, Nx), device=u.device, dtype=u.dtype)
        b = F.pad(b, (self.triL, 0, 0, 0, 0, 0))
        out[..., :self.modes] = self.quasi_diag_mul(b[..., :self.modes+self.bandwidth-1], self.weights)

        u = self.T.inv(out, self.X_dims)
        return u


class SOL1d(nn.Module):
    def __init__(self, T, in_channels, modes, width, bandwidth, out_channels=1, dim=1, skip=True, triL=0):
        super().__init__()

        self.modes = modes
        self.width = width
        self.triL = triL
        self.T = T
        self.dim = dim

        self.conv0 = PseudoSpectra1d(T, width, width, modes, bandwidth, triL)
        self.conv1 = PseudoSpectra1d(T, width, width, modes, bandwidth, triL)
        self.conv2 = PseudoSpectra1d(T, width, width, modes, bandwidth, triL)
        self.conv3 = PseudoSpectra1d(T, width, width, modes, bandwidth, triL)

        self.convl = PseudoSpectra1d(T, in_channels, width-in_channels, modes, bandwidth)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.skip = nn.Identity() if skip else ZerosFilling()

    def forward(self, x):

        # [batch, XYZ, c] -> [batch, c, XYZ]
        x = x.permute(0, -1, 1)

        x = torch.cat([x, F.gelu(self.convl(x))], dim=1)

        x = self.skip(x) + F.gelu(self.w0(x) + self.conv0(x))

        x = self.skip(x) + F.gelu(self.w1(x) + self.conv1(x))

        x = self.skip(x) + F.gelu(self.w2(x) + self.conv2(x))

        x = self.skip(x) + F.gelu(self.w3(x) + self.conv3(x))

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        x = self.T(self.T.inv(x, -2), -2)

        return x
