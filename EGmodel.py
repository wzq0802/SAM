import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from build_sam import sam_model_registry
import torch.nn.functional as F
import warnings
from torchvision import transforms
from math import log



class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)



class EdgeFusion(nn.Module):
    def __init__(self, channel=256):
        super(EdgeFusion, self).__init__()
        # 分支1: 边缘特征强化路径（单通道边缘图引导）
        self.edge_branch = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=3, padding=1),  # 将单通道边缘图映射到256通道
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            ConvBNR(channel, channel, 3)  # 进一步非线性变换
        )

        # 分支2: 原始特征保留路径
        self.orig_branch = nn.Sequential(
            ConvBNR(channel, channel, 3),
            nn.Conv2d(channel, channel, 3, padding=1)
        )

        # 动态融合门控（空间注意力）
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1),  # 压缩到单通道空间权重
            nn.Sigmoid()
        )

        # 通道注意力（与原设计一致）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_emb, edge):
        # Step 1: 反转边缘语义（1 - edge）并调整尺寸
        edge_mask = 1 - edge  # 高值=边界，低值=背景
        if edge_mask.size()[2:] != img_emb.size()[2:]:
            edge_mask = F.interpolate(edge_mask, img_emb.size()[2:], mode='bilinear', align_corners=False)

        # Step 2: 边缘分支处理（单通道->多通道）
        edge_feat = self.edge_branch(edge_mask)  # [B, 256, H, W]

        # Step 3: 原始特征分支处理
        orig_feat = self.orig_branch(img_emb)  # [B, 256, H, W]

        # Step 4: 动态门控融合
        gate_weight = self.fusion_gate(edge_feat)  # [B, 1, H, W]
        fused_feat = gate_weight * edge_feat + (1 - gate_weight) * orig_feat

        # Step 5: 通道注意力
        wei = self.avg_pool(fused_feat)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        output = fused_feat * wei

        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return out.view(x.size(0), x.size(1), 1, 1)

class Edge(nn.Module):
    def __init__(self):
        super(Edge, self).__init__()

        # 64 -> 128
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.ca1 = ChannelAttention(128)
        self.dilated1 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.relu1 = nn.ReLU(inplace=True)

        # 128 -> 256
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.ca2 = ChannelAttention(64)
        self.dilated2 = nn.Conv2d(64, 32, kernel_size=3, padding=4, dilation=4)
        self.relu2 = nn.ReLU(inplace=True)

        # 输出
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = x * self.ca1(x)
        x = self.relu1(self.dilated1(x))

        x = self.up2(x)
        x = x * self.ca2(x)
        x = self.relu2(self.dilated2(x))

        x = self.out_conv(x)
        edge_map = torch.sigmoid(x)
        return edge_map

# 定义 RockSAM 模型
class RockSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super(RockSAM, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.edge_head = Edge()  # 边缘检测任务头
        self.edge_fusion = EdgeFusion()
        # self.conv1 = nn.Conv2d(1,256,1)



        # 冻结 image_encoder 和 prompt_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        # 提取图像嵌入
        with torch.no_grad():  # 确保 image_encoder 的前向传播不计算梯度
            image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64)

        # 提取边缘特征
        edge = self.edge_head(image_embeddings)  # (B, 1, 64, 64)


        # 特征融合
        fused_embeddings = self.edge_fusion(image_embeddings, edge)  # (B, 256, 64, 64)


        # 调用 mask_decoder
        dense_prompt_embeddings = torch.zeros_like(fused_embeddings)


        sparse_prompt_embeddings = torch.zeros((fused_embeddings.size(0), 0, 256), device=fused_embeddings.device)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=fused_embeddings,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_prompt_embeddings,  # (B, 0, 256)
            dense_prompt_embeddings=dense_prompt_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )

        high_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return high_res_masks, edge



