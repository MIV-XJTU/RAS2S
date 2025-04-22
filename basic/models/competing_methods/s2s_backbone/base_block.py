import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import math
from .model_util import *
from einops import rearrange


class HFComp(nn.Module):
    def __init__(self, dim=24, num_heads=2, bias=True):
        super(HFComp, self).__init__()
        self.num_heads = num_heads

        self.k = nn.Conv2d(dim, dim, kernel_size=1,padding=0, bias=bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, padding=0,bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1,padding=0, bias=bias)

        self.res = make_layer(ResidualBlockNoBN, 2, mid_channels=dim)

    def forward(self, x, y):
        b, c, h, w = x.shape
        k = self.k(y)
        v = self.v(y)
        q = self.q(x)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.res(out)
        out = out+y

        return out

class SpecClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpecClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim
    def forward(self,guide_input):
        b,c,h,w, = guide_input.shape
        guide_input = guide_input.permute(0, 2, 3, 1).contiguous().view(b, -1, c)
        guide_feature_spec = self.fc1(guide_input)
        guide_feature_spec = guide_feature_spec.view(b, h, w, self.output_dim).permute(0, 3, 1, 2)
        return guide_feature_spec

class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out

class RouteFuncMLP(nn.Module):
    """
    The routing function for generating the calibration weights.
    """

    def __init__(self, c_in, ratio=1, kernels=[3,3], bn_eps=1e-5, bn_mmt=0.1):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
            kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(RouteFuncMLP, self).__init__()
        self.c_in = c_in
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.a = nn.Conv2d(
            in_channels=c_in,
            out_channels=int(c_in // ratio),
            kernel_size=[kernels[0], 1],
            padding=[kernels[0] // 2, 0],
        )
        self.relu = nn.ReLU(inplace=True)
        self.b = nn.Conv2d(
            in_channels=int(c_in // ratio),
            out_channels=c_in,
            kernel_size=[kernels[1], 1],
            padding=[kernels[1] // 2, 0],
            bias=False
        )
        self.b.skip_init = True
        self.b.weight.data.zero_()  # to make sure the initial values


    def forward(self, x):
        x = self.avgpool(x)
        x = self.a(x)

        x = self.relu(x)
        x = self.b(x) + 1
        return x

class SpecAdaConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 cal_dim="cin"):
        super(SpecAdaConv2d, self).__init__()

        assert cal_dim in ["cin", "cout"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cal_dim = cal_dim

        # base weights (W_b)
        self.weight = nn.Parameter(
            torch.Tensor(1, out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, alpha):

        _,  c_out, c_in, kh, kw = self.weight.size()
        b, c_in, h, w = x.size()
        x = x.reshape(1, -1, h, w)


        if self.cal_dim == "cin":
            weight = (alpha.unsqueeze(dim=1)* self.weight).reshape(-1, c_in // self.groups,kh, kw)
        elif self.cal_dim == "cout":
            weight = (alpha* self.weight).reshape(-1, c_in // self.groups, kh, kw)

        bias = None
        if self.bias is not None:
            bias = self.bias.repeat(b, 1).reshape(-1)
        output = F.conv2d(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*b)

        output = output.view(b,  c_out, output.size(-2), output.size(-1))

        return output

    def __repr__(self):
        return f"SpecAdaConv2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, " + \
               f"stride={self.stride}, padding={self.padding}, bias={self.bias is not None}, cal_dim=\"{self.cal_dim}\")"

class SpecDynConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpecDynConv, self).__init__()
        self.b = SpecAdaConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.b_rf = RouteFuncMLP(c_in=in_channels)
    def forward(self,input):
        x = self.b(input, self.b_rf(input))
        return x
    
class ConvGRU_base(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU_base, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q
        return h

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=24, input_dim=24, num_layers = 1, is_backward=False):
        super(ConvGRU, self).__init__()
        self.is_backward = is_backward
        self.conv_gru = ConvGRU_base(hidden_dim=hidden_dim,input_dim=hidden_dim)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if self.is_backward:
            self.fusion = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)
    def forward(self, hidden_state, input, hidden_state_f=None):

        hidden_state = self.conv_gru(hidden_state,input)
        if self.is_backward and hidden_state_f is not None:
            hidden_state = self.lrelu(self.fusion(torch.cat([hidden_state, hidden_state_f], dim=1)))

        return hidden_state

class AAM(nn.Module):
    def __init__(self, dec_hid_dim):
        super().__init__()
        self.conv1 = nn.Conv3d(dec_hid_dim+1, dec_hid_dim, kernel_size=3, padding=1)

    def forward(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(dim=1)
        hidden = torch.cat((hidden,encoder_outputs),dim=1)
        map = self.conv1(hidden)
        map = torch.tanh(map)
        hidden = F.softmax(map, dim=1)
        context = torch.sum(hidden * encoder_outputs, dim=1)

        return context