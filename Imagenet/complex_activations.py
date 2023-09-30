import torch
from torch import nn
import torch.nn.functional as F



class CReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.relu(x.real).type(torch.complex64) + 1j * F.relu(x.imag).type(torch.complex64)

class CPReLU(nn.Module):
    def __init__(self, num_channels= 1):
        super().__init__()
        self.num_channels= num_channels
        self.real_prelu = nn.PReLU(num_parameters= self.num_channels)
        self.imag_prelu = nn.PReLU(num_parameters= self.num_channels)

    def forward(self, x):
        return self.real_prelu(x.real).type(torch.complex64) + 1j * self.imag_prelu(x.imag).type(torch.complex64)

class zReLU(nn.Module):
    pass

class Naive_ComplexSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.sigmoid(x.real).type(torch.complex64) + 1j * F.sigmoid(x.imag).type(torch.complex64)

class Naive_ComplexTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.tanh(x.real).type(torch.complex64) + 1j * F.tanh(x.imag).type(torch.complex64)



