import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
import cv2
import numpy as np
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st
from network import Resnet
from my_functionals import GatedSpatialConv as gsc
from kan import KANLinear, KAN
from SETR import Setr_ConvFormer
from torch.nn import init



__all__ = ['KANLayer', 'KANBlock', 'KANBlock', 'DWConv', 'DW_bn_relu', 'PatchEmbed', 'ConvLayer', 'D_ConvLayer', 'UKAN']


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # # TODO
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)

        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=256, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        #self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)
        # 替换 encoder3 为 Setr_ConvFormer
        self.encoder3 = nn.Sequential(
            Setr_ConvFormer(
                n_channels=kan_input_dim // 4,  # 输入通道数
                n_classes=num_classes,  # 类别数
                imgsize=img_size // 4,  # 输入图像大小
                patch_num=16,  # patch 数量
                dim=embed_dims[0],  # 特征维度
                depth=depths[0],  # Transformer 深度
                heads=8,  # 多头注意力头数
                mlp_dim=embed_dims[0] * 4,  # MLP 维度
                dim_head=64,  # 每个注意力头的维度
                dropout=drop_rate,  # Dropout 率
                emb_dropout=drop_rate  # Embedding Dropout 率
            ),
            nn.Conv2d(embed_dims[0], kan_input_dim, kernel_size=1)  # 调整输出通道数为 kan_input_dim
        )
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)

        # 边缘检测分支
        self.dsn3 = nn.Conv2d(embed_dims[0], 1, 1)
        self.dsn4 = nn.Conv2d(embed_dims[1], 1, 1)
        self.dsn7 = nn.Conv2d(embed_dims[2], 1, 1)

        self.res1 = Resnet.BasicBlock(embed_dims[0] // 8, embed_dims[0] // 8, stride=1, downsample=None)
        self.d1 = nn.Conv2d(embed_dims[0] // 8, 32, 1)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)



        self.fusion_module = FusionModule(embed_dims, out_channels=embed_dims[0] // 8)

        self.sigmoid = nn.Sigmoid()

        #self.soft = nn.Softmax(dim=1)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t4)

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))
        out = self.final(out)

        ### 边缘检测分支
        s3 = F.interpolate(self.dsn3(t3), x.size()[2:], mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(t4), x.size()[2:], mode='bilinear', align_corners=True)
        #s7 = F.interpolate(self.dsn7(out), x.size()[2:], mode='bilinear', align_corners=True)

        m1f = F.interpolate(t1, x.size()[2:], mode='bilinear', align_corners=True)

        # im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        # canny = np.zeros((x.size(0), 1, x.size(2), x.size(3)))
        # for i in range(x.size(0)):
        #     canny[i] = cv2.Canny(im_arr[i], 10, 100)
        # canny = torch.from_numpy(canny).cuda().float()
        im_arr = x.cpu().numpy().transpose((0, 2, 3, 1)).astype(np.uint8)
        log_edges = np.zeros((x.size(0), 1, x.size(2), x.size(3)))
        sigma = 1.0

        for i in range(x.size(0)):
            # 对每个通道的图像进行高斯模糊，然后计算拉普拉斯
            blurred = cv2.GaussianBlur(im_arr[i], (5, 5), sigma)  # 5x5卷积核
            # 初始化log_edge为float64类型，以便存储float64的拉普拉斯结果
            log_edge = np.zeros_like(blurred[:, :, 0], dtype=np.float64)  # 使用float64类型

            for c in range(blurred.shape[2]):  # 对每个通道进行操作
                log_edge += cv2.Laplacian(blurred[:, :, c], cv2.CV_64F)  # 累加每个通道的拉普拉斯

            # 将单通道的拉普拉斯结果存储到log_edges中
            log_edges[i, 0] = log_edge

        # 转换为torch张量并移动到GPU
        log_edges = torch.from_numpy(log_edges).cuda().float()



        cs = self.res1(m1f)
        cs = F.interpolate(cs, x.size()[2:], mode='bilinear', align_corners=True)
        cs = self.d1(cs)
        cs = self.gate1(cs, s3)
        cs = self.res2(cs)
        cs = F.interpolate(cs, x.size()[2:], mode='bilinear', align_corners=True)
        cs = self.d2(cs)
        cs = self.gate2(cs, s4)
        cs = self.res3(cs)
        cs = F.interpolate(cs, x.size()[2:], mode='bilinear', align_corners=True)
        cs = self.d3(cs)
        #cs = self.gate3(cs, s7)
        cs = self.fuse(cs)
        cs = F.interpolate(cs, x.size()[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)

        cat = torch.cat((edge_out, log_edges), dim=1)
        acts = self.cw(cat)
        acts = self.sigmoid(acts)

        ### 融合模块
        fused_feature = self.fusion_module(t4, acts)  # 将分割特征与边缘特征融合
        out = out + fused_feature  # 将融合后的特征添加到分割特征中
        ### 最终输出
        out = self.final(out)
        return out, edge_out
        ########return out

class ASPP(nn.Module):
    def __init__(self, embed_dims=[256, 320, 512], out_channels=256, rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        # 1x1 卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(embed_dims[1], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 不同膨胀率的 3x3 空洞卷积
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(embed_dims[1], out_channels, kernel_size=3, padding=rates[0], dilation=rates[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(embed_dims[1], out_channels, kernel_size=3, padding=rates[1], dilation=rates[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3x3_3 = nn.Sequential(
            nn.Conv2d(embed_dims[1], out_channels, kernel_size=3, padding=rates[2], dilation=rates[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 图像级别的全局平均池化
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dims[1], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 特征融合的 1x1 卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 获取输入特征图的大小
        x_size = x.size()[2:]
        # 1x1 卷积
        conv1x1 = self.conv1x1(x)
        # 不同膨胀率的 3x3 空洞卷积
        conv3x3_1 = self.conv3x3_1(x)
        conv3x3_2 = self.conv3x3_2(x)
        conv3x3_3 = self.conv3x3_3(x)

        # 全局平均池化
        global_avg_pool = self.global_avg_pool(x)
        global_avg_pool = F.interpolate(global_avg_pool, size=x_size, mode='bilinear', align_corners=True)

        # 拼接所有特征
        out = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_avg_pool], dim=1)

        # 特征融合
        out = self.fusion_conv(out)

        return out

class FusionModule(nn.Module):
    def __init__(self, embed_dims, out_channels):
        super(FusionModule, self).__init__()

        # ASPP 模块
        self.aspp = ASPP(embed_dims, out_channels)

        # 边缘特征融合
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 特征拼接后的卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, seg_feature, edge_feature):
        """
        seg_feature: 分割分支的特征，形状为 (B, C, H, W)
        edge_feature: 边缘检测分支的输出，形状为 (B, 1, H, W)
        """
        # 使用 ASPP 模块提取多尺度特征
        aspp_feature = self.aspp(seg_feature)
        #print("aspp_feature shape:", aspp_feature.shape)  # 打印 aspp_feature 的尺寸
        # 将 aspp_feature 上采样到 edge_feature 的尺寸
        aspp_feature = F.interpolate(aspp_feature, size=edge_feature.shape[2:], mode='bilinear', align_corners=True)
        # 调整边缘特征的通道数
        edge_feature = self.edge_conv(edge_feature)
        #print("edge_feature shape:", edge_feature.shape)  # 打印 edge_feature 的尺寸
        # 将 edge_feature 下采样到 aspp_feature 的尺寸
        #edge_feature = F.interpolate(edge_feature, size=aspp_feature.shape[2:], mode='bilinear', align_corners=True)
        # 拼接 ASPP 特征和边缘特征
        fused_feature = torch.cat([aspp_feature, edge_feature], dim=1)

        # 最终卷积
        fused_feature = self.final_conv(fused_feature)

        return fused_feature