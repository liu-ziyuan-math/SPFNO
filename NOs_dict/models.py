"""
This code is written by Ziyuan Liu, you may contact us through liuziyuan17@nudt.edu.cn
"""
import os

from NOs_dict.SOL import *
from utilities import *

from torch.utils.data import DataLoader
from timeit import default_timer
from copy import deepcopy
import h5py
from scipy.io import loadmat
import fourierpack as sp
import chebypack as ch
from functools import partial as PARTIAL
from math import ceil

import matplotlib

_dst = PARTIAL(sp.Wrapper, [sp.sin_transform])
_idst = PARTIAL(sp.Wrapper, [sp.isin_transform])
_dct = PARTIAL(sp.Wrapper, [sp.cos_transform])
_idct = PARTIAL(sp.Wrapper, [sp.icos_transform])
_dcht = PARTIAL(ch.Wrapper, [ch.dct])
_idcht = PARTIAL(ch.Wrapper, [ch.idct])

_idctII = PARTIAL(sp.Wrapper, [sp.idctII])
_dctII = PARTIAL(sp.Wrapper, [sp.dctII])
_idstII = PARTIAL(sp.Wrapper, [sp.idstII])
_dstII = PARTIAL(sp.Wrapper, [sp.dstII])

DST = Transform(_dst, _idst)
DCT = Transform(_dct, _idct)
DST_II = Transform(_dstII, _idstII)
DCT_II = Transform(_dctII, _idctII)

# T_pipe = functools.partial(sp.FuncMat_Wrapper, [[sp.WSWA], [sp.sin_transform]])
# iT_pipe = functools.partial(sp.FuncMat_Wrapper, [[sp.iWSWA], [sp.isin_transform]])

SinNO = PARTIAL(SOL, DST)
CosNO = PARTIAL(SOL, DCT)
SinNO_II = PARTIAL(SOL, DST_II)
CosNO_II = PARTIAL(SOL, DCT_II)

SinNO1d = PARTIAL(SOL1d, DST)
CosNO1d = PARTIAL(SOL1d, DCT)
SinNO2d = PARTIAL(SOL, DST, dim=2)
CosNO2d = PARTIAL(SOL, DCT, dim=2)

# class SinNO(SOL):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=None, triL=0, skip=True):
#         super().__init__(DST, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)

# class CosNO(SOL):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=None, triL=0, skip=True):
#         super().__init__(DCT, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)
#
# class SinNO_II(SOL):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=None, triL=0, skip=True):
#         super().__init__(DST_II, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)
#
# class CosNO_II(SOL):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=None, triL=0, skip=True):
#         super().__init__(DCT_II, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)

# class SinNO2d(SinNO):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, triL=0, skip=True):
#         super().__init__(in_channels, modes, width, bandwidth, out_channels, dim=2, triL=triL, skip=skip)
#
# class CosNO2d(CosNO):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, triL=0, skip=True):
#         super().__init__(in_channels, modes, width, bandwidth, out_channels, dim=2, triL=triL, skip=skip)



class OPNO_neumann(SOL):
    def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=None, triL=0, skip=True):
        _dcht = functools.partial(ch.Wrapper, [ch.dct])
        _ishen_neumann = functools.partial(ch.Wrapper, [ch.icmp_neumann, ch.idct])
        Shen_neumann = Transform(_dcht, _ishen_neumann)
        super().__init__(Shen_neumann, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)

class OPNO_dirichlet(SOL):
    def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=None, triL=0, skip=True):
        _dcht = functools.partial(ch.Wrapper, [ch.dct])
        _ishen_dirichlet = functools.partial(ch.Wrapper, [ch.icmp_dirichlet, ch.idct])
        Shen_dirichlet = Transform(_dcht, _ishen_dirichlet)
        super().__init__(Shen_dirichlet, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)
    def forward(self, x):
        compst = torch.zeros_like(x[0, ..., 0:1])
        for i in self.X_dims:
            uR, uL = x[0, ..., 0].index_select(i, torch.tensor([0])), \
                     x[0, ..., 0].index_select(i, torch.tensor([-1]))
            grid = x[0, ..., 1:]
            compst += (uL+uR)/2 + (uR-uL)/2 * grid.repeat([1]*(-i))
        super().forward(self, x)
        return x+compst

class OPNO_robin(SOL):
    def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=None, triL=0, skip=True):
        _dcht = functools.partial(ch.Wrapper, [ch.dct])
        _ishen_robin = functools.partial(ch.Wrapper, [ch.icmp_robin, ch.idct])
        Shen_robin = Transform(_dcht, _ishen_robin)
        super().__init__(Shen_robin, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)


# class OPNO_neumann2d(OPNO_neumann):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, triL=0, skip=True):
#         super().__init__(in_channels, modes, width, bandwidth, out_channels, dim=2, triL=triL, skip=skip)
#
# class OPNO_dirichlet2d(OPNO_dirichlet):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, triL=0, skip=True):
#         super().__init__(in_channels, modes, width, bandwidth, out_channels, dim=2, triL=triL, skip=skip)
#
# class OPNO_robin2d(OPNO_robin):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, triL=0, skip=True):
#         super().__init__(in_channels, modes, width, bandwidth, out_channels, dim=2, triL=triL, skip=skip)

# class SinNO1d(SOL1d):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=1, triL=0, skip=True):
#         super().__init__(DST, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)
# class CosNO1d(SOL1d):
#     def __init__(self, in_channels, modes, width, bandwidth, out_channels=1, dim=1, triL=0, skip=True):
#         super().__init__(DCT, in_channels, modes, width, bandwidth, out_channels, dim, triL, skip)





# class WSWANO(nn.Module):
#     def __init__(self, in_channels, out_channels, degree1, degree2, bw1, bw2):
#         super(WSWANO, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.degree1 = degree1
#         self.degree2 = degree2
#         self.bw1, self.bw2 = bw1, bw2
#
#         self.scale = 2 / (in_channels + out_channels)
#         self.weights = nn.Parameter(
#             self.scale * torch.rand(in_channels*bw1*bw2, out_channels, degree1*degree2, dtype=torch.float32))
#         # self.weights = nn.Parameter(
#         #     self.scale * torch.rand(in_channels*bandwidth*bandwidth, out_channels, degree1*degree2, dtype=torch.complex64))
#
#         # self.unfold = torch.nn.Unfold(kernel_size=(self.bandwidth,self.bandwidth), padding=(self.bandwidth-1)//2)
#         self.unfold = torch.nn.Unfold(kernel_size=(bw1, bw2))
#
#     def quasi_diag_mul2d(self, input, weights):
#         xpad = self.unfold(input)
#         # print(xpad.shape, input.shape, weights.shape)
#         return torch.einsum("bix, iox->box", xpad, weights)
#         # return torch.einsum("bixw, xiow->box", xpad, weights)
#
#     def forward(self, u):
#         batch_size, width, Nx, Ny = u.shape
#
#         a = T_pipe(u, [-2, -1])
#
#         b = torch.zeros(batch_size, self.out_channels, Nx, Ny, device=u.device, dtype=torch.float32)
#         b[..., :self.degree1, :self.degree2] = \
#             self.quasi_diag_mul2d(a[..., :self.degree1+self.bw1-1, :self.degree2+self.bw2-1], self.weights).reshape(
#                 batch_size, self.out_channels, self.degree1, self.degree2)
#
#         u = iT_pipe(b, [-2, -1])
#         return u