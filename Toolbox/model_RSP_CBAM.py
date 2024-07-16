import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                #nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()
        # 使用指定数量的通道初始化卷积层
        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        # 应用第一个卷积和ReLU激活
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        # 应用第二个卷积
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        # 输入和卷积结果的逐元素加法
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs


# -------------DSFusion_start----------------------------------------
# SAM: Spatial Attention Module
class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=3, in_channels=32):
        super(SpatialAttention1, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels // 4, 1, kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        return self.sigmoid(x)

# SCAM: Channel Attention Module
class ChannelAttention1(nn.Module):
    def __init__(self, in_planes=32, ratio=4):
        super(ChannelAttention1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# -------------DSFusion_end----------------------------------------

# # -------------CBAM Start----------------------------------------
# class ChannelAttention(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(ChannelAttention, self).__init__()
#         self.maxpool = nn.AdaptiveMaxPool2d(1)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.se = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(channel // reduction, channel, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_result = self.maxpool(x)
#         avg_result = self.avgpool(x)
#         max_out = self.se(max_result)
#         avg_out = self.se(avg_result)
#         output = self.sigmoid(max_out + avg_out)
#         return output

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         max_result, _ = torch.max(x, dim=1, keepdim=True)
#         avg_result = torch.mean(x, dim=1, keepdim=True)
#         result = torch.cat([max_result, avg_result], 1)
#         output = self.conv(result)
#         output = self.sigmoid(output)
#         return output

# class CBAMBlock(nn.Module):
#     def __init__(self, channel=32, reduction=16, kernel_size=7):
#         super(CBAMBlock, self).__init__()
#         self.ca = ChannelAttention(channel=channel, reduction=reduction)
#         self.sa = SpatialAttention(kernel_size=kernel_size)

#     def forward(self, x):
#         residual = x
#         out = x * self.ca(x)
#         out = out * self.sa(out)
#         return out + residual

# # -------------CBAM End----------------------------------------

# -----------------------------------------------------
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        channel = 32 # 定义通道数
        spectral_num = 8 # 定义光谱图像的通道数

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        
        # 定义第一个卷积层，将光谱图像的通道数转换为内部通道数
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        
        # 定义四个残差块
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()

# -------------DSFusion_start-------------
        self.spatial_attention = SpatialAttention1(in_channels=channel)
        self.channel_attention = ChannelAttention1(in_planes=channel)
# -------------DSFusion_end---------------

        # self.cbam = CBAMBlock(channel)# 定义CBAM块

        # 定义最后一个卷积层，将内部通道数转换回光谱图像的通道数
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        # 将四个残差块串联成一个序列
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

        # 初始化权重，这一步非常重要！
        init_weights(self.backbone, self.conv1, self.conv3)   # state initialization, important!

    def forward(self, x, y):  # x= hp of ms; y = hp of pan

        # 将全色图像的通道复制八次，以匹配多光谱图像的通道数
        pan_concat = torch.cat([y, y, y, y, y, y, y, y], 1)  # Bsx8x64x64

        # 计算全色图像和多光谱图像高频部分的差异
        input = torch.sub(pan_concat, x)  # Bsx8x64x64

        # 应用第一个卷积层和ReLU激活函数
        rs = self.relu(self.conv1(input))  # Bsx32x64x64
        # # 应用CBAM模块
        # rs = self.cbam(rs)
        # 通过残差块序列处理
        rs = self.backbone(rs)  # ResNet's backbone!

# -------------DSFusion_start-------------
        sa = self.spatial_attention(rs)  # Apply spatial attention
        rs = rs * sa

        ca = self.channel_attention(rs)  # Apply channel attention
        rs = rs * ca
# -------------DSFusion_end---------------
        
        

        # 应用最后一个卷积层，得到输出
        output = self.conv3(rs)  # Bsx8x64x64

        return output

# ----------------- End-Main-Part ------------------------------------
#方差缩放:初始权重
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor

#打印模型的摘要或参数
def summaries(model, writer=None, grad=False):
    
    # 如果grad参数为True，则打印模型的摘要
    if grad:
        from torchsummary import summary

        # 打印模型的摘要，指定输入尺寸和批次大小
        summary(model, input_size=[(8, 64, 64), (1, 64, 64)], batch_size=1)
    else:

        # 如果grad参数为False，则遍历模型的所有参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    # 如果提供了writer对象，则使用它来添加模型图
    if writer is not None:
        x = torch.randn(1, 64, 64, 64)
        writer.add_graph(model,(x,))