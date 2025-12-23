"""
使用Ultralytics框架的二值化卷积模块
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv
from .block import C2f


"""
计算激活层的可偏移参数
用于第一阶段
"""
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
"""
激活二值化函数
用于第一阶段
"""
class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

"""
仅激活二值化的卷积
用于第一阶段，内置LearnableBias偏移模块
"""
class BinaryConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__(c1, c2, k, s, p, g, d, act=nn.Identity())
        self.bias = nn.Parameter(torch.zeros(1, c2, 1, 1), requires_grad=True)  # 可学习偏置
        self.act = BinaryActivation()

    def forward(self, x):
        # LearnableBias: x + bias
        out = x + self.bias.expand_as(x)
        # Conv + BN + BinaryActivation
        return self.act(self.bn(self.conv(out)))



"""
权重和激活都二值化的卷积
用于第二阶段
"""
class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


"""
C2f二值化版本
用于第一阶段，仅激活二值化
"""
class BinaryC2f(C2f):
    def __init__(self, c1: int, c2: int = 0, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize binary C2f layer with BinaryConv activations."""
        # 处理 c2=0 的情况（兼容省略 c2 的 yaml 配置）
        if c2 == 0:
            c2 = c1
        super().__init__(c1, c2, n, shortcut, g, e)
        # 将内部的 Conv 替换为 BinaryConv（仅 cv1 和 cv2）
        self.cv1 = BinaryConv(c1, 2 * self.c, 1, 1)
        self.cv2 = BinaryConv((2 + n) * self.c, c2, 1)
    
