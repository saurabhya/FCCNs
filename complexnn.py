from typing import TypeVar, Union, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F


"""

write a test file to check the working of 1j * (tensor) 

This implementation is actually correct and it is an easier way to do things.

"""


T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, tuple[T]]
_scalar_or_tuple_2_t = Union[T, tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, tuple[T, T, T]]

# For arguments which respresnt size parameters (eg. kernerl_size, padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]


def apply_complex(fr, fi, input, dtype= torch.complex64):
    return (fr(input.real) - fi(input.imag)).type(dtype) + 1j * (fr(input.imag) + fi(input.real)).type(dtype)


class ComplexConv2d(nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t= 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t= 1,
        groups:int = 1,
        bias: bool = False,
        complex_axis= 1,
        padding_mode: str = 'zeros',
        device= None,
        dtype= None
        ) -> None:
        super().__init__()

        # # check condition that the in_channels are even
        # if (in_channels % 2 != 0) or (out_channels % 2 != 0):
        #     raise ValueError(f"in_channels and out_channels should be even. Got {in_channels} and {out_channels}") 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, groups= groups, bias= bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size= kernel_size, stride= stride, padding= padding, dilation= dilation, groups= groups, bias= bias)

        # weight init is not being used here
        # although I am including the code here which can be later used to initialize weights
        # nn.init.normal_(self.conv_real.weight.data, std= 0.05)
        # nn.init.normal_(self.conv_imag.weight.data, std= 0.05)
        # if bias:
        #     nn.init.constant_(self.conv_real.bias, 0.0)
        #     nn.init.constant_(self.conv_imag.bias, 0.0)


    def forward(self, x):
        ''' define how the forward prop will take place '''
        # check if the input is of dtype complex
        # for this we can use is_complex() function which will return true if the input is complex dtype
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")

        return apply_complex(self.conv_real, self.conv_imag, x)


class ComplexTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t= 1,
        padding: _size_2_t= 0,
        output_padding: _size_2_t= 0,
        groups: int= 1,
        bias: bool= False,
        dilation: int= 1,
        padding_mode: str= 'zeros',
        device= None,
        dtype= None
    ):
        super().__init__()

        self.trans_conv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride= stride, padding= padding, output_padding= output_padding, groups= groups, bias= bias, dilation= dilation)
        self.trans_conv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride= stride, padding= padding, output_padding= output_padding, groups= groups, bias= bias, dilation= dilation)

        # nn.init.normal_(self.trans_conv_real.weight.data, std= 0.05)
        # nn.init.normal_(self.trans_conv_imag.weight.data, std= 0.05)
        # if bias:
        #     nn.init.constant_(self.trans_conv_real.bias, 0.0)
        #     nn.init.constant_(self.trans_conv_imag.bias, 0.0)

    def forward(self, x):
        ''' define how the forward prop will take place '''
        # check if the input is of dtype complex
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")

        return apply_complex(self.trans_conv_real, self.trans_conv_imag, x)

# At wavefrontshaping's implementation I saw the implmentation of various complex pooling methods,
# so let's try to define that and may be we can use it later.

class ComplexMaxPool2d(nn.Module):
    """
    The implementation in wavefrontshaping's code says that for max pooling we will take into account the absolute value and phase.

    But we will skip that and try to implement as the standard way i.e.
    apply max pooling on real and imaginary parts separately.
    
    Since max pooling layer does not have any additional parameter, we can use only one
    pooling layer for both components.

    (Please verify that from docs)

    **Verified**

    """
    def __init__(self, kernel_size, stride= None, padding= 0, dilation= 1, return_indices= False, ceil_mode= False):
        super().__init__()

        self.kernel_size= kernel_size
        self.stride = stride
        self.padding= padding
        self.dilation= dilation
        self.ceil_mode= ceil_mode
        self.return_indices= return_indices

        self.max_pool = nn.MaxPool2d(self.kernel_size, self.stride, self.padding, self.dilation, self.return_indices, self.ceil_mode)

    def forward(self, x):

        # check if the input is complex
        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor, Got {x.dtype}")

        return (self.max_pool(x.real)).type(torch.complex64) + 1j * (self.max_pool(x.imag)).type(torch.complex64)


########################################################################
# BEGIN HERE TOMORROW


class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride= None, padding= 0, ceil_mode= False, count_include_pad= True, divisor_override= None):
        super().__init__()

        self.kernel_size= kernel_size
        self.stride= stride
        self.padding= padding
        self.ceil_mode= ceil_mode
        self.count_include_pad= count_include_pad
        self.divisor_override= divisor_override

        self.avg_pool = nn.AvgPool2d(self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)


    def forward(self, x):
        # check is x is complex tensor

        if not x.is_complex():
            raise ValueError(f"Input should be a complex tensor. Got {x.dtype}")

        
        return (self.avg_pool(x.real)).type(torch.complex64) + 1j * (self.avg_pool(x.imag)).type(torch.complex64)

class ComplexAdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.output_size= output_size

        self.adaptive_pool= nn.AdaptiveAvgPool2d(self.output_size)

    def forward(self, input):
        return (self.adaptive_pool(input.real)).type(torch.complex64) + 1j * (self.adaptive_pool(input.imag)).type(torch.complex64)



class ComplexDropout(nn.Module):
    def __init__(self, p= 0.5):
        super().__init__()
        self.p = p
        self.real_drop = nn.Dropout(self.p)
        self.imag_drop = nn.Dropout(self.p)

    def forward(self, input):
        # if self.training:
        #     mask = torch.ones(*input.shape, dtype= torch.float32)
        #     mask = F.dropout(mask, self.p, self.training) * 1/(1-self.p)
        #     mask = mask.type(input.type)
        #
        #     return mask * input
        #
        # else:
        #     return input

        return (self.real_drop(input.real)).type(torch.complex64) + 1j * (self.imag_drop(input.imag)).type(torch.complex64)



class ComplexDropout2d(nn.Module):
    def __init__(self, p= 0.5):
        super(ComplexDropout2d).__init__()
        self.p= p


    def forward(self, input):
        if self.training:
            mask = torch.ones(*input.shape, dtype= torch.float32)
            mask = F.dropout2d(mask, self.p, self.training) * 1/(1-self.p)
            mask = mask.type(input.dtype)

            return mask * input

        else:
            return input


class ComplexNaiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps= 1e-05, momentum=0.1, affine= True, track_running_stats= True, device= None):
        super().__init__()

        self.num_features= num_features
        self.eps= eps
        self.momentum= momentum
        self.affine= affine
        self.track_running_stats= track_running_stats
        self.device= device

        self.real_bn = nn.BatchNorm2d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)
        self.imag_bn = nn.BatchNorm2d(self.num_features, self.eps, self.momentum, self.affine, self.track_running_stats)

        

    def forward(self, input):
        # check if the input is a complex tensor
        if not input.is_complex():
            raise ValueError(f"Input should be complex, Got {input.dtype}")
        
        return (self.real_bn(input.real)).type(torch.complex64) + 1j * (self.imag_bn(input.imag)).type(torch.complex64)




# class ComplexBatchNorm2d(nn.Module):
#     def __init__(self, num_features, eps= 1e-05, momentum= 0.1, affine= True, track_running_stats= True):
#         super().__init__()
#
#         self.num_features= num_features // 2
#         self.eps= eps
#         self.momentum= momentum
#         self.affine= affine
#         self.track_running_stats= track_running_stats
#
#         if self.affine:
#             self.Wrr = torch.nn.parameter.Parameter(torch.Tensor(self.num_features))
#             self.Wri = torch.nn.parameter.Parameter(torch.Tensor(self.num_features))
#             self.Wii= torch.nn.parameter.Parameter(torch.Tensor(self.num_features))
#             self.Br = torch.nn.parameter.Parameter(torch.Tensor(self.num_features))
#             self.Bi = torch.nn.parameter.Parameter(torch.Tensor(self.num_features))
#         else:
#             self.register_parameter('Wrr', None)
#             self.register_parameter('Wri', None)
#             self.register_parameter('Wii', None)
#             self.register_parameter('Br', None)
#             self.register_parameter('Bi', None)
#
#         if self.track_running_stats:
#             self.register_buffer('RMr', torch.zeros(self.num_features))
#             self.register_buffer('RMi', torch.zeros(self.num_features))
#             self.register_buffer('RVrr', torch.zeros(self.num_features))
#             self.register_buffer('RVri', torch.zeros(self.num_features))
#             self.register_buffer('RVii', torch.zeros(self.num_features))
#             self.register_buffer('num_batches_tracked', torch.tensor(0, dtype= torch.long))
#
#         else:
#             self.register_parameter('RMr', None)
#             self.register_parameter('RMi', None)
#             self.register_parameter('RVrr', None)
#             self.register_parameter('RVri', None)
#             self.register_parameter('RVii', None)
#             self.register_parameter('num_batches_tracked', None)
#
#         self.reset_parameters()
#
#     def reset_running_stats(self):
#         if self.track_running_stats:
#             self.RMr.zero_()
#             self.RMi.zero_()
#             self.RVrr.fill_(1)
#             self.RVri.zero_()
#             self.RVii.fill_(1)
#             self.num_batches_tracked.zero_()
#
#     def reset_parameters(self):
#         self.reset_running_stats()
#         if self.affine:
#             self.Br.data.zero_()
#             self.Bi.data.zero_()
#             self.Wrr.data.fill_(1)
#             self.Wri.data.uniform_(-.9, +.9) # W will be positive-ddefinite
#             self.Wii.data.fill_(1)
#
#     def _check_input_dim(self, xr, xi):
#         assert(xr.shappe == xi.shape)
#         assert(xr.size(1) == self.num_features)
#
#
#     def forward(self, input):
#
#         xr, xi = input.real, input.imag
#         exponential_average_factor = 0.0
#
#         if self.training and self.track_running_stats:
#             self.num_batches_tracked += 1
#             if self.momentum is None:
#                 exponential_average_factor = 1.0 / self.num_batches_tracked.item()
#             else:
#                 exponential_average_factor = self.momentum
#
#         training = self.training or not self.track_running_stats
#         redux = [i for i in reversed(range(xr.dim())) if i != 1]
#         vdim = [1] * xr.dim()
#         vdim[1] = xr.size(1)
#
#         if training:
#             Mr, Mi = xr, xi
#             for d in redux:
#                 Mr = Mr.mean(d, keepdim= True)
#                 Mi = Mi.mean(d, keepdim= True)
#
#             if self.track_running_stats:
#                 self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
#                 self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
#
#         else:
#             Mr = self.RMr.view(vdim)
#             Mi = self.RMi.view(vdim)
#
#         xr, xi = xr-Mr, xr-Mi
#
#         if training:
#             Vrr = xr * xr
#             Vri = xr * xi
#             Vii = xi * xi
#
#             for d in redux:
#                 Vrr = Vrr.mean(d, keepdim= True)
#                 Vri = Vri.mean(d, keepdim= True)
#                 Vii = Vii.mean(d, keepdim= True)
#
#             if self.track_running_stats:
#                 self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
#                 self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
#                 self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
#         else:
#             Vrr = self.RVrr.view(vdim)
#             Vri = self.RVri.view(vdim)
#             Vii = self.RVii.view(vdim)
#
#         Vrr = Vrr + self.eps
#         Vri = Vri
#         Vii = Vii + self.eps
#
#
#         tau = Vrr + Vii
#         # delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
#         delta = torch.addcmul(Vrr * Vii, Vri, Vri, value= -1)
#         s = delta.sqrt()
#         t = (tau + 2*s).sqrt()
#
#
#         rst= (s * t).reciprocal()
#         Urr= (s + Vii) * rst
#         Uii= (s + Vrr) * rst
#         Uri= (- Vri) * rst
#
#
#         if self.affine:
#             Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
#             Zrr = (Wrr * Urr) + (Wri * Uri)
#             Zri = (Wrr * Uri) + (Wri * Uii)
#             Zir = (Wri * Urr) + (Wii * Uri)
#             Zii = (Wri * Uri) + (Wii * Uii)
#
#         else:
#             Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii
#
#         yr = (Zrr * xr) + (Zri * xi)
#         yi = (Zir * xr) + (Zii * xi)
#
#         if self.affine:
#             yr = yr + self.Br.view(vdim)
#             yi = yi + self.Bi.view(vdim)
#
#         return (yr).type(torch.complex64) + 1j* (yi).type(torch.complex64)
#
#     def extra_repr(self):
#         return '{num_features}, eps= {eps}, momentum= {momentum}, affine= {affine},'\
#                 'track_running_stats= {track_running_stats}'.format(**self.__dict__)
#


class ComplexBatchNorm2d(torch.nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
            track_running_stats=True, complex_axis=1):
        super().__init__()
        self.num_features        = num_features
        self.eps                 = eps
        self.momentum            = momentum
        self.affine              = affine
        self.track_running_stats = track_running_stats

        self.complex_axis = complex_axis

        if self.affine:
            self.Wrr = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Br  = torch.nn.Parameter(torch.Tensor(self.num_features))
            self.Bi  = torch.nn.Parameter(torch.Tensor(self.num_features))
        else:
            self.register_parameter('Wrr', None)
            self.register_parameter('Wri', None)
            self.register_parameter('Wii', None)
            self.register_parameter('Br',  None)
            self.register_parameter('Bi',  None)

        if self.track_running_stats:
            self.register_buffer('RMr',  torch.zeros(self.num_features))
            self.register_buffer('RMi',  torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones (self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones (self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('RMr',                 None)
            self.register_parameter('RMi',                 None)
            self.register_parameter('RVrr',                None)
            self.register_parameter('RVri',                None)
            self.register_parameter('RVii',                None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9) # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, inputs):
        #self._check_input_dim(xr, xi)

        # xr, xi = torch.chunk(inputs,2, axis=self.complex_axis)
        xr, xi = inputs.real, inputs.imag
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        #
        # NOTE: The precise meaning of the "training flag" is:
        #       True:  Normalize using batch   statistics, update running statistics
        #              if they are being collected.
        #       False: Normalize using running statistics, ignore batch   statistics.
        #
        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i!=1]
        vdim  = [1] * xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)
        xr, xi = xr-Mr, xi-Mi

        #
        # Variance Matrix V Computation
        #
        # Includes epsilon numerical stabilizer/Tikhonov regularizer.
        # Includes running variance update if training and running.
        #
        if training:
            Vrr = xr * xr
            Vri = xr * xi
            Vii = xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)
        Vrr   = Vrr + self.eps
        Vri   = Vri
        Vii   = Vii + self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        # sqrt of a 2x2 matrix,
        # - https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau   = Vrr + Vii
        # delta = torch.addcmul(Vrr * Vii, -1, Vri, Vri)
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value= -1)
        s     = delta.sqrt()
        t     = (tau + 2*s).sqrt()

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        rst   = (s * t).reciprocal()
        Urr   = (s + Vii) * rst
        Uii   = (s + Vrr) * rst
        Uri   = (  - Vri) * rst

        #
        # Optionally left-multiply U by affine weights W to produce combined
        # weights Z, left-multiply the inputs by Z, then optionally bias them.
        #
        # y = Zx + B
        # y = WUx + B
        # y = [Wrr Wri][Urr Uri] [xr] + [Br]
        #     [Wir Wii][Uir Uii] [xi]   [Bi]
        #
        if self.affine:
            Wrr, Wri, Wii = self.Wrr.view(vdim), self.Wri.view(vdim), self.Wii.view(vdim)
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        return (yr).type(torch.complex64) + 1j * (yi).type(torch.complex64)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
                'track_running_stats={track_running_stats}'.format(**self.__dict__)
