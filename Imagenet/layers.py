import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# th.use_deterministic_algorithms(True)


def Cmul(x, y):
    """
    Complex multiplication of two complex vectors.
    x: Tensor of shape [B, 2, C, H, W]
    y: Tensor of shape [B, 2, C, H, W]
    """
    a, b = x[:, 0], x[:, 1]
    c, d = y[:, 0], y[:, 1]

    real = (a*c - b*d)
    imag = (b*c + a*d)

    return th.stack([real, imag], dim=1)


def Cdiv(x, y, clamp=False):
    """
    Complex division of two complex vectors.
    x: Tensor of shape [B, 2, C, H, W]
    y: Tensor of shape [B, 2, C, H, W]
    clamp: Clamp the denominator to be non-zero, instead of adding a small value.
    """

    a, b = x[:, 0], x[:, 1]
    c, d = y[:, 0], y[:, 1]

    real = (a*c - b*d)
    imag = (b*c + a*d)

    if clamp:
        divisor = th.clamp(c**2 + d**2, 0.05)
    else:
        divisor = c**2 + d**2 + 1e-7

    real = (a*c + b*d)/divisor  # ac + bd
    imag = (b*c - a*d)/divisor  # (bc - ad)i

    return th.stack([real, imag], dim=1)


def Cconj(x):
    """
    Complex conjugate of a complex vector.
    x: Tensor of shape [B, 2, C, H, W]
    """
    a, b = x[:, 0], x[:, 1]
    return th.stack([a, -b], dim=1)


def abs_normalize(w):
    """
    Normalize the weights so that the sum of the absolute values is 1.
    """
    return w/(w.detach().abs().sum(dim=(1, 2, 3), keepdim=True)+1e-6)


def normalize(w):
    """
    Normalize the weights so that the sum is 1.
    """
    return w/(w.sum(dim=(1, 2, 3), keepdim=True)+1e-6)


def sq_normalize(w):
    """
    Normalize the weights so that the sum of the squares is 1.
    """
    w_sq = w**2
    return w_sq/(w_sq.sum(dim=(1, 2, 3), keepdim=True)+1e-6)


def retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=2)
    output = flattened_tensor.gather(
        dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
    return output


class reflect_pad(nn.Module):
    """
    Does 2D reflection padding (size 1), but deterministically.
    In some pytorch versions, the existing kernel is not deterministic.
    """

    def __init__(self) -> None:
        super(reflect_pad, self).__init__()

    def forward(self, x):
        shape = x.shape
        assert len(shape) == 4
        to_return = th.zeros(
            (shape[0], shape[1], shape[2]+2, shape[3]+2), device=x.device, dtype=x.dtype)
        to_return[..., 1:-1, 1:-1] = to_return[..., 1:-1, 1:-1] + x
        to_return[..., 0] = to_return[..., 0] + to_return[..., 1]
        to_return[..., -1] = to_return[..., -1] + to_return[..., -2]
        to_return[..., 0, :] = to_return[..., 0, :] + to_return[..., 1, :]
        to_return[..., -1, :] = to_return[..., -1, :] + to_return[..., -2, :]
        return to_return


class MaxPoolMag(nn.Module):
    """ 
    Performs magnitude max pooling
    Pools the input with largest magnitude over neighbors
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
        super(MaxPoolMag, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size, stride,
                               padding, dilation, True, ceil_mode)

    def __repr__(self):
        return 'ComplexMagnitudePooling'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        x_norm = th.norm(x, dim=1)
        _, indices = self.mp(x_norm)
        x_real = retrieve_elements_from_indices(x[:, 0, ...], indices)
        x_imag = retrieve_elements_from_indices(x[:, 1, ...], indices)
        return th.cat((x_real.unsqueeze(1), x_imag.unsqueeze(1)), dim=1)


class ComplexConv(nn.Module):
    # Our complex convolution implementation
    def __init__(self, in_channels, num_filters, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, bias=False, new_init=False, use_groups_init=False, fan_in=False, *args, **kwargs):
        super(ComplexConv, self).__init__()

        # Convolution parameters
        self.in_channels = in_channels
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.reflect = reflect
        if reflect:
            self.pad_func = th.nn.ZeroPad2d(reflect)

        self.A = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)
        self.B = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)

        if new_init:
            if fan_in:
                c = in_channels
            else:
                c = num_filters
            if use_groups_init:
                c = c/groups

            gain = 1/np.sqrt(2)
            with th.no_grad():
                std = gain / np.sqrt(kern_size * kern_size * c)
                self.A.weight.normal_(0, std)
                self.B.weight.normal_(0, std)

    def __repr__(self):
        return 'ComplexConv'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        if len(x.shape) == 5:
            N, CC, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x.reshape(N*CC, C, H, W))
            x = x.reshape(N, CC, C, x.shape[-2], x.shape[-1])
            real = x[:, 0]
            imag = x[:, 1]
            out_real = self.A(real) - self.B(imag)
            out_imag = self.B(real) + self.A(imag)
            return th.stack([out_real, out_imag], dim=1)
        else:
            N, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x)
            out_real = self.A(x)
            out_imag = self.B(x)
            return th.stack([out_real, out_imag], dim=1)


class ComplexConvFast(nn.Module):
    # Our complex convolution implementation
    def __init__(self, in_channels, num_filters, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, bias=False, new_init=False, use_groups_init=False, fan_in=False, *args, **kwargs):
        super(ComplexConvFast, self).__init__()

        # Convolution parameters
        self.in_channels = in_channels
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.reflect = reflect
        if reflect:
            self.pad_func = th.nn.ZeroPad2d(reflect)

        self.A = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)
        self.B = nn.Conv2d(in_channels, num_filters, kern_size,
                           stride=stride, padding=padding, groups=groups, bias=bias)

        if new_init:
            if fan_in:
                c = in_channels
            else:
                c = num_filters
            if use_groups_init:
                c = c/groups

            gain = 1/np.sqrt(2)
            with th.no_grad():
                std = gain / np.sqrt(kern_size * kern_size * c)
                self.A.weight.normal_(0, std)
                self.B.weight.normal_(0, std)

    def __repr__(self):
        return 'ComplexConv'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        if len(x.shape) == 5:
            N, CC, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x.reshape(N*CC, C, H, W))
            x = x.reshape(N, CC, C, x.shape[-2], x.shape[-1])
            real = x[:, 0]
            imag = x[:, 1]
            t1 = self.A(real)
            t2 = self.B(imag)

            t3 = F.conv2d(real+imag, weight=(self.A.weight + self.B.weight), stride=self.stride,
                          padding=self.padding, groups=self.groups)

            return th.stack([t1 - t2, t3 - t1 - t2], dim=1)
        else:
            N, C, H, W = x.shape
            if self.reflect:
                x = self.pad_func(x)
            out_real = self.A(x)
            out_imag = self.B(x)
            return th.stack([out_real, out_imag], dim=1)


class NaiveCBN(nn.Module):
    """
    Naive BatchNorm which concatenates real and imaginary channels
    """

    def __init__(self, channels):
        super(NaiveCBN, self).__init__()
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        x_shape = x.shape
        return self.bn(x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])).reshape(x_shape)


class VNCBN(nn.Module):
    """
    Equivariant Complex Batch Norm
    Computes magnitude of the complex input and applies batch norm on it
    """

    def __init__(self, channels):
        super(VNCBN, self).__init__()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        mag = th.norm(x, dim=1)
        normalized = self.bn(mag)
        mag_factor = normalized/(mag+1e-6)
        return x*mag_factor[:, None, ...]


class DivLayer(nn.Module):
    """
    division layer
    """

    def __init__(self, in_channels, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, use_one_filter=True, new_init=False):
        super(DivLayer, self).__init__()

        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.reflect = reflect
        self.use_one_filter = use_one_filter

        if self.use_one_filter:
            self.conv = ComplexConv(
                in_channels, 1, kern_size, stride, padding, dilation, groups, reflect=reflect, new_init=new_init)
        else:
            self.conv = ComplexConv(in_channels, in_channels, kern_size,
                                    stride, padding, dilation, groups, reflect=reflect, new_init=new_init)

    def __repr__(self):
        return 'DivLayer'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W

        y = x

        if self.use_one_filter:
            conv = self.conv(y)
            conv = conv.repeat(1, 1, C, 1, 1)
        else:
            conv = self.conv(y)

        # For center-cropping original input
        output_xdim = conv.shape[-2]
        output_ydim = conv.shape[-1]
        input_xdim = H
        input_ydim = W

        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)

        num = x[:, :, :, start_x:start_x +
                output_xdim, start_y:start_y+output_ydim]

        a, b = num[:, 0], num[:, 1]
        c, d = conv[:, 0], conv[:, 1]

        divisor = c**2 + d**2 + 1e-7

        real = (a*c + b*d)/divisor  # ac + bd
        imag = (b*c - a*d)/divisor  # (bc - ad)i

        return th.stack([real, imag], dim=1)


class ConjugateLayer(nn.Module):
    """
    conjugate layer
    """

    def __init__(self, in_channels, kern_size, stride=(1, 1), padding=0, dilation=1, groups=1, reflect=False, use_one_filter=False, new_init=False):
        super(ConjugateLayer, self).__init__()
        self.kern_size = kern_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.reflect = reflect
        self.use_one_filter = use_one_filter
        conv = ComplexConv

        if self.use_one_filter:
            self.conv = conv(
                in_channels, 1, kern_size, stride, padding, dilation, groups, reflect=reflect, new_init=new_init)
        else:
            self.conv = conv(in_channels, in_channels, kern_size,
                             stride, padding, dilation, groups, reflect=reflect, new_init=new_init)

    def __repr__(self):
        return 'Conjugate'

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W
        y = x

        if self.use_one_filter:
            conv = self.conv(y)
            conv = conv.repeat(1, 1, C, 1, 1)
        else:
            conv = self.conv(y)

        # For center-cropping original input
        output_xdim = conv.shape[-2]
        output_ydim = conv.shape[-1]
        input_xdim = H
        input_ydim = W

        start_x = int((input_xdim-output_xdim)/2)
        start_y = int((input_ydim-output_ydim)/2)

        num = x[:, :, :, start_x:start_x +
                output_xdim, start_y:start_y+output_ydim]

        a, b = num[:, 0], num[:, 1]
        c, d = conv[:, 0], conv[:, 1]
        real = (a*c + b*d)  # ac + bd
        imag = (b*c - a*d)  # (bc - ad)i

        x = th.stack([real, imag], dim=1)

        return x


class DistFeatures(nn.Module):
    """
    prototype distance layer, using Euclidean distance
    """

    def __init__(self, in_channels, num_prototypes=16):
        super(DistFeatures, self).__init__()

        # Convolution parameters
        self.in_channels = in_channels
        self.num_prototypes = num_prototypes

        prototypes = th.rand(2, in_channels, num_prototypes)
        self.prototypes = nn.Parameter(data=prototypes, requires_grad=True)
        self.temp = nn.Parameter(data=th.tensor(1.0), requires_grad=True)

    def __repr__(self):
        return 'DistFeats'

    def forward(self, x, y=None):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C = x.shape

        if y is not None:
            y = y[..., 0, 0]
            a, b = self.prototypes[None, 0], self.prototypes[None, 1]
            c, d = y[:, 0, ..., None], y[:, 1, ..., None]
            real = a*c - b*d
            imag = b*c + a*d
        else:
            prototypes = self.prototypes
            real, imag = prototypes[None, 0, :, :], prototypes[None, 1, :, :]
        a, b = x[:, 0, :, None], x[:, 1, :, None]

        dist_sq = (real-a)**2 + (imag-b)**2
        dist = th.sqrt(dist_sq.mean(dim=1))

        return -dist*self.temp


class scaling_layer(nn.Module):
    """
    scaling layer for GTReLU
    """

    def __init__(self, channels, g_global=False):
        super(scaling_layer, self).__init__()
        if g_global:
            channels = 1
        self.a_bias = nn.Parameter(th.rand(channels,), requires_grad=True)
        self.b_bias = nn.Parameter(th.rand(channels,), requires_grad=True)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W
        x_c = x[:, 0]
        x_d = x[:, 1]

        a_bias = self.a_bias[None, :, None, None]
        b_bias = self.b_bias[None, :, None, None]

        real_component = a_bias * x_c - b_bias * x_d
        imag_component = b_bias * x_c + a_bias * x_d

        return th.stack([real_component, imag_component], dim=1)


class Two_Channel_Nonlinearity(th.autograd.Function):
    """
    Non-linearity which thresholds phase
    """

    @staticmethod
    def forward(ctx, inputs):
        temp_phase = inputs

        phase_mask = temp_phase % (2*np.pi)
        phase_mask = (phase_mask <= np.pi).type(
            th.cuda.FloatTensor) * (phase_mask >= 0).type(th.cuda.FloatTensor)
        temp_phase = temp_phase * phase_mask

        ctx.save_for_backward(inputs, phase_mask)

        return temp_phase

    @staticmethod
    def backward(ctx, grad_output):
        inputs, phase_mask = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = grad_input*(1-phase_mask)

        return grad_input


class eqnl(nn.Module):
    """
    Equivariant version of the phase-only tangent ReLU
    """

    def __init__(self, channels, trelu_b=0.0, *args, **kwargs):
        # Applies tangent reLU to inputs.
        super(eqnl, self).__init__()
        self.phase_scale = nn.Parameter(
            th.ones(channels,), requires_grad=True)
        self.cn = Two_Channel_Nonlinearity.apply
        self.trelu_b = trelu_b

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        p1 = x
        p2 = th.mean(x, dim=2, keepdim=True)

        abs1 = th.norm(p1, dim=1, keepdim=True) + 1e-6
        abs2 = th.norm(p2, dim=1, keepdim=True) + 1e-6
        p2 = p2/abs2

        conjp2 = th.stack((p2[:, 0], -p2[:, 1]), 1)
        shifted = Cmul(p1, conjp2)
        phasediff = th.atan2(
            shifted[:, 1], shifted[:, 0] + (shifted[:, 0] == 0) * 1e-5)
        final_phase = self.cn(phasediff) * \
            th.relu(self.phase_scale[None, :, None, None])

        out = abs1 * \
            Cmul(th.stack([th.cos(final_phase), th.sin(final_phase)], 1), p2)

        return out


class GTReLU(nn.Module):
    """
    GTReLU layer
    """

    def __init__(self, channels, g_global=False, phase_scale=False):
        super(GTReLU, self).__init__()
        self.ps = phase_scale
        if g_global:
            channels = 1
        self.a_bias = nn.Parameter(th.rand(channels,), requires_grad=True)
        self.b_bias = nn.Parameter(th.rand(channels,), requires_grad=True)

        self.relu = nn.ReLU()
        self.cn = Two_Channel_Nonlinearity.apply
        if self.ps:
            self.phase_scale = nn.Parameter(
                th.ones(channels,)*phase_scale)

    def forward(self, x):
        """
        x: Tensor of shape [B, 2, C, H, W]
        """
        N, CC, C, H, W = x.shape  # Batch, 2, channels, H, W
        x_c = x[:, 0]
        x_d = x[:, 1]

        # Scaling
        a_bias = self.a_bias[None, :, None, None]
        b_bias = self.b_bias[None, :, None, None]

        real_component = a_bias * x_c - b_bias * x_d
        imag_component = b_bias * x_c + a_bias * x_d

        x = th.stack([real_component, imag_component], dim=1)

        # Thresholding
        temp_abs = th.norm(x, dim=1)
        temp_phase = th.atan2(
            x[:, 1, ...], x[:, 0, ...] + (x[:, 0, ...] == 0) * 1e-5)

        final_abs = temp_abs.unsqueeze(1)

        final_phase = self.cn(temp_phase)

        x = th.cat((final_abs * th.cos(final_phase).unsqueeze(1),
                   final_abs * th.sin(final_phase).unsqueeze(1)), 1)

        # Phase scaling [Optional]
        if self.ps:
            norm = th.norm(x, dim=1)
            angle = th.atan2(x[:, 1], x[:, 0] +
                             (x[:, 0] == 0) * 1e-5)
            angle = angle * \
                th.minimum(th.maximum(self.phase_scale[None, :, None, None],
                                      th.tensor(0.5)), th.tensor(2.0))

            x = th.stack([norm*th.cos(angle), norm*th.sin(angle)], dim=1)

        return x


class DCN_CBN(th.nn.Module):
    """
    DCN's CBN
    Mostly based on Pytorch th/nn/modules/batchnorm.py and DCN's Complex BatchNorm implementation
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, weight_init=None, bias_init=None):
        super(DCN_CBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.Wrr = th.nn.Parameter(th.Tensor(num_features))
            self.Wri = th.nn.Parameter(th.Tensor(num_features))
            self.Wii = th.nn.Parameter(th.Tensor(num_features))
            self.Br = th.nn.Parameter(th.Tensor(num_features))
            self.Bi = th.nn.Parameter(th.Tensor(num_features))
        else:
            pass
        if self.track_running_stats:
            self.register_buffer('RMr',  th.zeros(num_features))
            self.register_buffer('RMi',  th.zeros(num_features))
            self.register_buffer('RVrr', th.ones(num_features))
            self.register_buffer('RVri', th.zeros(num_features))
            self.register_buffer('RVii', th.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 th.tensor(0, dtype=th.long))
        else:
            pass
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr .zero_()
            self.RMi .zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br .data.zero_()
            self.Bi .data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)  # W will be positive-definite
            self.Wii.data.fill_(1)

    def _check_input_dim(self, xr, xi):
        assert(xr.shape == xi.shape)
        assert(xr.size(1) == self.num_features)

    def forward(self, xr, xi):
        self._check_input_dim(xr, xi)

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
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1]*xr.dim()
        vdim[1] = xr.size(1)

        #
        # Mean M Computation and Centering
        #
        # Includes running mean update if training and running.
        #
        if training:
            Mr = xr
            Mi = xi
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
            Vrr = xr*xr
            Vri = xr*xi
            Vii = xi*xi
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
        Vrr = Vrr+self.eps
        Vri = Vri
        Vii = Vii+self.eps

        #
        # Matrix Inverse Square Root U = V^-0.5
        #
        tau = Vrr+Vii
        delta = th.addcmul(Vrr*Vii, -1, Vri, Vri)
        s = delta.sqrt()
        t = (tau + 2*s).sqrt()
        rst = (s*t).reciprocal()

        Urr = (s+Vii)*rst
        Uii = (s+Vrr)*rst
        Uri = (-Vri)*rst

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
            Zrr = self.Wrr.view(Urr.shape)*Urr + self.Wri.view(Uri.shape)*Uri
            Zri = self.Wrr.view(Uri.shape)*Uri + self.Wri.view(Uii.shape)*Uii
            Zir = self.Wri.view(Urr.shape)*Urr + self.Wii.view(Uri.shape)*Uri
            Zii = self.Wri.view(Uri.shape)*Uri + self.Wii.view(Uii.shape)*Uii
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr, yi = Zrr*xr + Zri*xi, Zir*xr + Zii*xi

        if self.affine:
            yr = yr + self.Br[None, :, None, None]
            yi = yi + self.Bi[None, :, None, None]

        return yr, yi

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(
                   **self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        super(DCN_CBN, self)._load_from_state_dict(state_dict,
                                                   prefix,
                                                   local_metadata,
                                                   strict,
                                                   missing_keys,
                                                   unexpected_keys,
                                                   error_msgs)


class ComplexBN(th.nn.Module):
    """Wrapper around DCN_CBN"""

    def __init__(self, *args, **kwargs):
        super(ComplexBN, self).__init__()
        self.BN = DCN_CBN(*args, **kwargs)

    def forward(self, x):
        return th.stack(self.BN(x[:, 0, ...], x[:, 1, ...]), dim=1)
