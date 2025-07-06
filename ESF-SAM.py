import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from build_sam import sam_model_registry
import warnings
from torchvision import transforms
from math import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import cast
from torch import Tensor

warnings.filterwarnings("ignore")
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

data_path = os.listdir("data1/ori")
data_path.sort(key=lambda x: int(x[:-4]))
data_path1 = os.listdir("data1/seg")
data_path1.sort(key=lambda x: int(x[:-4]))
data_path2 = os.listdir("data1/edge")
data_path2.sort(key=lambda x: int(x[:-4]))

train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]

def default_loader(path):
    data_pil = np.load(f"EG-SAM/data1/ori/{path}").reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor) / 255.0  
    data_tensor = data_tensor.repeat(3, 1, 1)

    transform = transforms.Resize((1024, 1024))
    data_tensor = transform(data_tensor)
    return data_tensor


def default_loader1(path):
    data_pil1 = np.load(f"EG-SAM/data1/seg/{path}").reshape((1, 256, 256))
    data_tensor1 = torch.tensor(data_pil1).type(torch.FloatTensor)
    return data_tensor1


def default_loader2(path):
    data_pil2 = np.load(f"EG-SAM/data1/edge/{path}").reshape((1, 256, 256))
    data_tensor2 = torch.tensor(data_pil2).type(torch.FloatTensor)
    return data_tensor2

class NpyDataset(Dataset):
    def __init__(self, paths, loader=default_loader, loader1=default_loader1, loader2=default_loader2):
        self.images = paths
        self.loader = loader
        self.loader1 = loader1
        self.loader2 = loader2

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.loader1(fn)
        edge = self.loader2(fn)
        return img, target, edge

    def __len__(self):
        return len(self.images)


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

        self.edge_branch = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=3, padding=1),            
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            ConvBNR(channel, channel, 3)
        )

        self.orig_branch = nn.Sequential(
            ConvBNR(channel, channel, 3),
            nn.Conv2d(channel, channel, 3, padding=1)
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1),             
            nn.Sigmoid()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img_emb, edge):

        edge_mask = 1 - edge
        if edge_mask.size()[2:] != img_emb.size()[2:]:
            edge_mask = F.interpolate(edge_mask, img_emb.size()[2:], mode='bilinear', align_corners=False)

        edge_feat = self.edge_branch(edge_mask)

        orig_feat = self.orig_branch(img_emb)


        gate_weight = self.fusion_gate(edge_feat)
        fused_feat = gate_weight * edge_feat + (1 - gate_weight) * orig_feat

 
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

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.ca1 = ChannelAttention(128)
        self.dilated1 = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.ca2 = ChannelAttention(64)
        self.dilated2 = nn.Conv2d(64, 32, kernel_size=3, padding=4, dilation=4)
        self.relu2 = nn.ReLU(inplace=True)


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

class RockSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super(RockSAM, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.edge_head = Edge()
        self.edge_fusion = EdgeFusion()

        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):

        with torch.no_grad():
            image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64)


        edge = self.edge_head(image_embeddings)  # (B, 1, 64, 64)

        fused_embeddings = self.edge_fusion(image_embeddings, edge)  # (B, 256, 64, 64)

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


sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
rock_sam = RockSAM(
    image_encoder=sam.image_encoder,
    mask_decoder=sam.mask_decoder,
    prompt_encoder=sam.prompt_encoder,
).to(device)

for param in rock_sam.parameters():
    param.requires_grad = False
for param in rock_sam.mask_decoder.parameters():
    param.requires_grad = True
for param in rock_sam.edge_head.parameters():
    param.requires_grad = True
for param in rock_sam.edge_fusion.parameters():
    param.requires_grad = True


optimizer = optim.Adam(filter(lambda p: p.requires_grad, rock_sam.parameters()), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5, verbose=True)


bce_loss = nn.BCEWithLogitsLoss()
bce_loss1 = nn.BCELoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True).cuda()
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

from pytorch_msssim import ssim

def ssim_loss(pred, target):

    return 1 - ssim(pred.unsqueeze(1),
                    target.unsqueeze(1),
                    data_range=1.0,
                    size_average=True)

class FocalLoss(nn.Module):
    def __init__(self, edge_alpha=0.8, gamma=2.0):
        super().__init__()
        self.edge_alpha = edge_alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)

        focal_loss = torch.where(target == 0,
                               self.edge_alpha * (1 - pt) ** self.gamma * bce_loss,
                               (1 - self.edge_alpha) * (1 - pt) ** self.gamma * bce_loss)
        return focal_loss.mean()

awl = AutomaticWeightedLoss(num=2).to(device)
focal_loss = FocalLoss().to(device)


def loss(pred, target, edge_pred=None, edge_target=None):
    target_resized = F.interpolate(target, size=pred.shape[1:], mode="bilinear", align_corners=False)
    target_resized = target_resized.squeeze(1)  # Remove extra channel dimension
    loss_seg = bce_loss(pred, target_resized)

    if edge_pred is not None and edge_target is not None:
        edge_pred = edge_pred.squeeze(1)

        edge_target_resized = edge_pred.squeeze(1)
        loss_edge = 0.4*focal_loss(edge_pred, edge_target_resized) + 0.4*dice_loss(edge_pred, edge_target_resized)+0.2*ssim_loss(edge_pred, edge_target_resized)


        total_loss = awl(loss_seg, loss_edge)
        return total_loss

    return loss_seg


train_data = NpyDataset(train_paths)
val_data = NpyDataset(val_paths)

batch_size = 4
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)


epochs = 100
patience = 20
best_val_loss = float("inf")
early_stop_counter = 0

Loss_list = []
Val_loss_list = []

for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    rock_sam.train()


    for data, label, edge in tqdm(trainloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        data, label, edge = data.to(device), label.to(device), edge.to(device)

        pred, edge_pred = rock_sam(data)
        pred = pred.squeeze(1)


        loss_value = loss(pred, label, edge_pred, edge)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

    train_loss /= len(trainloader)
    Loss_list.append(train_loss)

    rock_sam.eval()
    with torch.no_grad():
        for data_val, label_val, edge_val in valloader:
            data_val, label_val, edge_val = data_val.to(device), label_val.to(device), edge_val.to(device)

            pred_val, edge_pred_val = rock_sam(data_val)
            pred_val = pred_val.squeeze(1)              
            val_loss += loss(pred_val, label_val, edge_pred_val, edge_val).item()

    val_loss /= len(valloader)
    Val_loss_list.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if (epoch + 1) % 50 == 0:
        torch.save(rock_sam.state_dict(), f"EG-SAM/new1/EG-SAM/rock_sam_epoch_{epoch+1}.pth")
        print(f"Model saved at Epoch {epoch + 1}")

   
    scheduler.step(val_loss)
