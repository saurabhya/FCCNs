import torch
import numpy as np
import torch.nn as nn
import layers
from functools import partial
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)


def count_params(model): return sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)


def convert_mag_cos_sin(inputs, eps=1e-9):
    # Converts a [B, C, H, W] input into a [B, 3, C, H, W] complex representation
    mag = torch.abs(inputs)
    phase = torch.angle(inputs)
    return torch.stack([torch.log(mag+eps), torch.cos(phase), torch.sin(phase)], dim=1)


class complex2real(nn.Module):
    def __init__(self):
        super(complex2real, self).__init__()

    def forward(self, x):
        return torch.stack([x.real, x.imag], dim=1)


class real2complex(nn.Module):
    def __init__(self):
        super(real2complex, self).__init__()

    def forward(self, x):
        return x[:, 0]+1j*x[:, 1]


class small_cnn(nn.Module):
    # Backbone cnn layer
    def __init__(self, groups=5, in_size=None, no_clutter=True, use_mag=False, *args, **kwargs):
        super(small_cnn, self).__init__()
        self.relu = nn.ReLU()
        self.use_mag = use_mag
        if in_size is None:
            if self.use_mag:
                self.conv_1 = nn.Conv2d(1, 30, (5, 5), groups=groups)
            else:
                self.conv_1 = nn.Conv2d(2, 30, (5, 5), groups=groups)
        else:
            self.conv_1 = nn.Conv2d(in_size, 30, (5, 5), groups=groups)

        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3), groups=groups)
        self.bn_1 = nn.GroupNorm(5, 30)
        self.bn_2 = nn.GroupNorm(10, 50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(60, 70, (2, 2), groups=groups)
        self.bn_3 = nn.GroupNorm(14, 70)
        self.linear_2 = nn.Linear(70, 30)
        out_size = 10 if no_clutter else 11
        self.linear_4 = nn.Linear(30, out_size)
        self.res1 = nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2 = nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))

    def make_res_block(self, in_channel, out_channel):
        res_block = []
        res_block.append(nn.GroupNorm(5, in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(
            out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.GroupNorm(5, int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4),
                         out_channel, (1, 1), bias=False))
        return res_block

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.mean(dim=(-1, -2))
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x


class CDS_I(nn.Module):
    """
    CDS Model (I-Type) for small CIFAR experiments. Based on CIFARnet
    """
    # Our model for the real-valued dataset experiments

    def __init__(self, cifarnet_config='dgtf', dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(CDS_I, self).__init__()
        print("CIFARnet Config:", cifarnet_config)
        self.cifarnet_config = cifarnet_config

        # Building layers....
        conv = layers.ComplexConv
        diff = layers.DivLayer

        inp_size = 2 if ((dset_type == 'lab') or (
            dset_type == 'sliding')) else 3
        self.wfm1 = conv(inp_size, 16, kern_size=3, stride=(
            2, 2), reflect=1, new_init=True, use_groups_init=True, bias=False)

        self.wfm2 = conv(16, 32, kern_size=3,
                         stride=(2, 2), reflect=1, groups=2, new_init=True, use_groups_init=True)
        self.wfm3 = conv(32, 64, kern_size=3,
                         stride=(2, 2), reflect=1, groups=4, new_init=True, use_groups_init=True)
        self.wfm4 = conv(64, 64, kern_size=4, groups=64,
                         new_init=True, use_groups_init=True)

        self.diff1 = diff(16, 3, reflect=1, new_init=True)

        self.gtrelu1 = layers.GTReLU(16, phase_scale=True)
        self.gtrelu2 = layers.GTReLU(32, phase_scale=True)
        self.gtrelu3 = layers.GTReLU(64, phase_scale=True)

        self.fc1 = conv(64, prototype_size, 1, groups=4, new_init=True)

        self.bn = nn.BatchNorm1d(prototype_size*2)

        dist_feat = layers.DistFeatures

        self.dist_feat = dist_feat(prototype_size, outsize)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)
        x = self.wfm3(x)
        x = self.gtrelu3(x)
        x = self.wfm4(x)

        x = self.fc1(x)
        x_shape = x.shape
        x = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape)
        x = self.dist_feat(x[..., 0, 0])

        return x


class CDS_E(nn.Module):
    """
    CDS Model (E-Type) for small CIFAR experiments. Based on CIFARnet
    """

    def __init__(self, dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
        super(CDS_E, self).__init__()

        conv = layers.ComplexConv
        inp_size = 2 if (dset_type == 'lab') else 3
        self.wfm1 = conv(inp_size, 16, kern_size=3, stride=(
            2, 2), reflect=1, new_init=True, use_groups_init=True)
        self.wfm2 = conv(16, 32, kern_size=3, stride=(
            2, 2), reflect=1, groups=2, new_init=True, use_groups_init=True)
        self.wfm3 = conv(32, 64, kern_size=3, stride=(
            2, 2), reflect=1, groups=4, new_init=True, use_groups_init=True)
        self.wfm4 = conv(64, 64, kern_size=4, groups=64,
                         new_init=True, use_groups_init=True)

        self.s1 = layers.scaling_layer(16)
        self.s2 = layers.scaling_layer(32)
        self.s3 = layers.scaling_layer(64)

        self.t1 = layers.eqnl(
            16, clampdiv=True, groups=1)
        self.t2 = layers.eqnl(
            32, clampdiv=True, groups=1)
        self.t3 = layers.eqnl(
            64, clampdiv=True, groups=1)

        self.fc1 = conv(64, prototype_size, 1, groups=4, new_init=True)

        df_conv = conv(prototype_size, prototype_size, kern_size=1, groups=16,
                       new_init=True, use_groups_init=True)

        self.dist_feat = layers.DistFeatures(prototype_size, outsize)

        self.bn = layers.VNCBN(prototype_size)

    def forward(self, x):
        # Convert complex input into a real-imaginary input
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.s1(x)
        x = self.t1(x)
        x = self.wfm2(x)
        x = self.s2(x)
        x = self.t2(x)
        x = self.wfm3(x)
        x = self.s3(x)
        x = self.t3(x)
        x = self.wfm4(x)

        x = self.fc1(x)

        x = self.bn(x)

        y = torch.sum(x, dim=2, keepdim=True)/np.sqrt(x.shape[2]*2)
        x = self.dist_feat(x[..., 0, 0], y)

        return x


class CDS_MSTAR(nn.Module):
    """
    MSTAR Model (I-Type) for MSTAR experiments
    """

    def __init__(self, no_clutter=True, *args, **kwargs):
        super(CDS_MSTAR, self).__init__()
        # for groups in cnn backbone
        groups = 5

        conv = layers.ComplexConv
        diff = layers.ConjugateLayer
        self.wfm1 = conv(1, 5, 5, (1, 1), groups=1)
        self.diff1 = diff(5, 3, groups=1)
        self.gtrelu1 = layers.GTReLU(5)
        self.mp = layers.MaxPoolMag(2)
        self.wfm2 = conv(5, 5, 3, (2, 2), groups=1)
        self.gtrelu2 = layers.GTReLU(5)
        self.cnn = small_cnn(groups=groups, no_clutter=no_clutter, in_size=15)

    def forward(self, x):
        x = torch.stack([x.real, x.imag], dim=1)
        x = self.wfm1(x)
        x = self.diff1(x)
        x = self.gtrelu1(x)
        x = self.mp(x)
        x = self.wfm2(x)
        x = self.gtrelu2(x)

        mag = torch.norm(x, dim=1)
        phase = torch.atan2(x[:, 1, ...], x[:, 0, ...])

        mag = mag + 1e-5
        log_mag = torch.log(mag)

        log_mag = log_mag.unsqueeze(1)

        cos = torch.cos(phase)
        cos = cos.unsqueeze(1)

        sin = torch.sin(phase)
        sin = sin.unsqueeze(1)
        x = torch.cat([log_mag, cos, sin], dim=1)

        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2],
                      x.shape[-2], x.shape[-1])
        return self.cnn(x)


def conv_bn_complex(c_in, c_out, groups=1):
    return nn.Sequential(
        layers.ComplexConvFast(c_in, c_out, kern_size=3,
                               padding=1, groups=groups),
        layers.ComplexBN(c_out),
        nn.ReLU(True),
    )


class residual_complex(nn.Module):
    def __init__(self, c, groups=1):
        super(residual_complex, self).__init__()
        self.res = nn.Sequential(
            conv_bn_complex(c, c, groups=groups),
            conv_bn_complex(c, c, groups=groups)
        )

    def forward(self, x):
        return x + self.res(x)


class flatten(nn.Module):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class mul(nn.Module):
    def __init__(self, c):
        super(mul, self).__init__()
        self.c = c

    def forward(self, x):
        return x * self.c


def CDS_large(outsize=10, *args, **kwargs):
    channels = {'prep': 64,
                'layer1': 128, 'layer2': 256, 'layer3': 256}
    n = [
        layers.ComplexConvFast(2, channels['prep'], kern_size=3, padding=1, groups=1),

        layers.ConjugateLayer(channels['prep'], kern_size=1, use_one_filter=True),

        conv_bn_complex(channels['prep'], channels['prep'], groups=2),
        conv_bn_complex(channels['prep'], channels['layer1'], groups=2),
        layers.MaxPoolMag(2),
        residual_complex(channels['layer1'], groups=2),
        conv_bn_complex(channels['layer1'], channels['layer2'], groups=4),
        layers.MaxPoolMag(2),
        conv_bn_complex(channels['layer2'], channels['layer3'], groups=2),
        layers.MaxPoolMag(2),
        residual_complex(channels['layer3'], groups=4),
        layers.MaxPoolMag(4),
        flatten(),
        nn.Linear(channels['layer3']*2, outsize, bias=False),
        mul(0.125),
    ]
    return nn.Sequential(*n)
