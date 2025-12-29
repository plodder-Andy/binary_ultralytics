"""
使用Ultralytics框架的二值化卷积模块
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv


"""
计算激活层的可偏移参数，即对输入添加一个可学习的偏置
通过创建一个形状为(1, out_chn, 1, 1)的参数张量，并在前向传播时将其扩展到与输入张量相同的形状，然后将其加到输入张量上。
"""
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
"""
通过输入激活层的可偏移参数，用于激活二值化
"""
class BinAct(nn.Module):
    def __init__(self):
        super(BinAct, self).__init__()

    def forward(self, x):
        # 得到前向传播的二值化输出
        out_forward = torch.sign(x) 

        # 使用分段函数近似梯度
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        
        # 直通估计器
        # 前向的二值输出 - 近似梯度的输出 + 近似梯度的输出
        out = out_forward.detach() - out3.detach() + out3

        return out    


"""
仅激活二值化的卷积，使用了偏移参数
"""
class BinActConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__(c1, c2, k, s, p, g, d, act=nn.Identity())
        self.c1 = c1  # 保存输入通道数
        self.c2 = c2  # 保存输出通道数
        self.bias = LearnableBias(c1)  
        self.act = BinAct()

    def forward(self, x):
        out = self.bias(x)                       # 为输入添加可学习的偏置
        return self.act(self.bn(self.conv(out))) # Conv + BN + BinAct


"""
权重二值化卷积
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
权重二值化卷积封装，包含批归一化和可偏移激活函数
"""
class BinWgtConv(nn.Module):
    default_act = nn.PReLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充
            act (bool | nn.Module): 激活函数
        """
        super().__init__()

        def autopad(k, p=None, d=1):
            if d > 1: k = d * (k - 1) + 1
            if p is None: p = k // 2
            return p

        p = autopad(k, p) if p is None else p
        self.binary_conv = HardBinaryConv(c1, c2, kernel_size=k, stride=s, padding=p)
        self.bias = LearnableBias(c1)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        out = self.bias(x)
        return self.act(self.bn(self.binary_conv(out)))

"""
LSQ激活量化模块，在激活函数后应用LSQ量化
支持按通道量化
"""
class LSQAct(nn.Module):
    def __init__(self, nbits=8, init_scale=1.0):
        super().__init__()
        self.nbits = nbits

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        self.Qp = 2 ** nbits - 1   # 量化上限，非对称量化
        self.grad_scale = 1.0

    def forward(self, x):
        # 计算梯度缩放因子（LSQ论文中的公式）
        if self.training:
            self.grad_scale = 1.0 / math.sqrt(self.Qp * x.numel())

        # 获取缩放因子，按通道量化需要调整形状
        s = self.scale

        # 量化
        x = x / s  # 归一化
        x = torch.clamp(x, 0, self.Qp)  # 截断
        x_floor = torch.round(x)  # 四舍五入

        # 直通估计器（STE）
        if self.training:
            # 前向传播使用量化值，反向传播使用近似梯度
            x_quant = x_floor.detach() - x.detach() + x
        else:
            x_quant = x_floor

        # 反量化
        x_dequant = x_quant * s

        return x_dequant

"""
权重二值化 + 激活8bit量化的卷积层 (w1a8)
使用LSQ进行激活量化
"""
class LSQConv(nn.Module):
    default_act = nn.ReLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, act=True, act_nbits=8):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充
            act (bool | nn.Module): 激活函数
            act_nbits (int): 激活量化位宽，默认8bit
        """
        super().__init__()

        def autopad(k, p=None, d=1):
            if d > 1: k = d * (k - 1) + 1
            if p is None: p = k // 2
            return p

        p = autopad(k, p) if p is None else p

        # 权重二值化卷积
        self.binary_conv = HardBinaryConv(c1, c2, kernel_size=k, stride=s, padding=p)

        # 可学习偏置（激活前）
        self.bias = LearnableBias(c1)

        # 批归一化
        self.bn = nn.BatchNorm2d(c2)

        # 激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # LSQ激活量化（8bit）
        self.lsq_activation = LSQAct(nbits=act_nbits, init_scale=1.0)

    def forward(self, x):
        out = self.bias(x)              # 输入偏置
        out = self.binary_conv(out)     # 二值化权重卷积
        out = self.bn(out)              # 批归一化
        out = self.act(out)             # 激活函数
        out = self.lsq_activation(out)  # 激活量化

        return out

"""
权重8bit LSQ量化模块
使用对称量化（权重有正负值）
"""
class LSQWgt(nn.Module):
    def __init__(self, nbits=8, init_scale=1.0):
        super().__init__()
        self.nbits = nbits

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        # 对称量化范围
        self.Qn = -2 ** (nbits - 1)      # 量化下限
        self.Qp = 2 ** (nbits - 1) - 1   # 量化上限

        self.grad_scale = 1.0

    def forward(self, weight):
        """
        对权重进行LSQ量化
        Args:
            weight: 浮点权重张量
        Returns:
            量化后的权重张量
        """
        # 计算梯度缩放因子
        if self.training:
            self.grad_scale = 1.0 / math.sqrt(self.Qp * weight.numel())

        # 量化
        s = self.scale
        w = weight / s  # 归一化
        w = torch.clamp(w, self.Qn, self.Qp)  # 截断到量化范围
        w_floor = torch.round(w)  # 四舍五入

        # 直通估计器（STE）
        if self.training:
            w_quant = w_floor.detach() - w.detach() + w
        else:
            w_quant = w_floor

        # 反量化
        w_dequant = w_quant * s

        return w_dequant

"""
权重8bit + 激活8bit量化的全精度卷积层 (w8a8)
使用LSQ进行权重和激活量化
用于第一层和最后一层
"""
class LSQFullConv(nn.Module):
    default_act = nn.ReLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, act=True, weight_nbits=8, act_nbits=8):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 卷积核大小
            s (int): 步长
            p (int, optional): 填充
            act (bool | nn.Module): 激活函数
            weight_nbits (int): 权重量化位宽，默认8bit
            act_nbits (int): 激活量化位宽，默认8bit
        """
        super().__init__()

        def autopad(k, p=None, d=1):
            if d > 1: k = d * (k - 1) + 1
            if p is None: p = k // 2
            return p

        p = autopad(k, p) if p is None else p

        # 全精度卷积层（权重将被量化）
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)

        # 权重量化模块（8bit对称量化）
        self.lsq_weight = LSQWgt(nbits=weight_nbits, init_scale=1.0)

        # 可学习偏置（激活前）
        self.bias = LearnableBias(c1)

        # 批归一化
        self.bn = nn.BatchNorm2d(c2)

        # 激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # 激活量化模块（8bit非对称量化）
        self.lsq_activation = LSQAct(nbits=act_nbits, init_scale=1.0)

    def forward(self, x):
        # 1. 输入偏置
        out = self.bias(x)

        # 2. 对权重进行LSQ量化，然后进行卷积
        quantized_weight = self.lsq_weight(self.conv.weight)
        out = F.conv2d(out, quantized_weight, stride=self.conv.stride,
                      padding=self.conv.padding, dilation=self.conv.dilation,
                      groups=self.conv.groups)

        # 3. 批归一化
        out = self.bn(out)

        # 4. 激活函数
        out = self.act(out)

        # 5. 激活量化
        out = self.lsq_activation(out)

        return out
    

