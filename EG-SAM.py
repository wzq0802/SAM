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

warnings.filterwarnings("ignore")

# 固定随机种子
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 数据路径
data_path = os.listdir("point/data/ori")
data_path.sort(key=lambda x: int(x[:-4]))
data_path1 = os.listdir("point/data/seg")
data_path1.sort(key=lambda x: int(x[:-4]))

# 训练验证划分
train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]

# 数据加载器
def default_loader(path):
    data_pil = np.load(f"point/data/ori/{path}").reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor) / 255.0  # 归一化到 [0, 1]

    # 将单通道灰度图像扩展为三通道
    data_tensor = data_tensor.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]

    # 调整到 SAM 模型期望的分辨率（1024x1024）
    transform = transforms.Resize((1024, 1024))
    data_tensor = transform(data_tensor)
    return data_tensor


def default_loader1(path):
    data_pil1 = np.load(f"point/data/seg/{path}").reshape((1, 256, 256))
    data_tensor1 = torch.tensor(data_pil1).type(torch.FloatTensor)
    return data_tensor1


# 自定义数据集
class NpyDataset(Dataset):
    def __init__(self, paths, loader=default_loader, loader1=default_loader1):
        self.images = paths
        self.loader = loader
        self.loader1 = loader1

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.loader1(fn)
        return img, target

    def __len__(self):
        return len(self.images)


# 定义 RockSAM 模型
class RockSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super(RockSAM, self).__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # 冻结 image_encoder 和 prompt_encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        # 提取图像嵌入
        with torch.no_grad():  # 确保 image_encoder 的前向传播不计算梯度
            image_embeddings = self.image_encoder(image)  # (B, 256, 64, 64)


        dense_prompt_embeddings = torch.zeros_like(image_embeddings)
        sparse_prompt_embeddings = torch.zeros((image_embeddings.size(0), 0, 256), device=image_embeddings.device)

        # 调用 mask_decoder
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
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

        return high_res_masks


# 加载 SAM 模型
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
rock_sam = RockSAM(
    image_encoder=sam.image_encoder,
    mask_decoder=sam.mask_decoder,
    prompt_encoder=sam.prompt_encoder,
).to(device)

# 确保只训练 mask_decoder 的参数
for param in rock_sam.parameters():
    param.requires_grad = False
for param in rock_sam.mask_decoder.parameters():
    param.requires_grad = True


optimizer = optim.Adam(filter(lambda p: p.requires_grad, rock_sam.parameters()), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5, verbose=True)

# 损失函数
bce_loss = nn.BCEWithLogitsLoss()


def loss(pred, target):
    target_resized = F.interpolate(target, size=pred.shape[1:], mode="bilinear", align_corners=False)
    # 移除 target 的多余通道维度
    target_resized = target_resized.squeeze(1)
    # 计算损失
    loss_bce = bce_loss(pred, target_resized)
    return loss_bce


# 数据加载
train_data = NpyDataset(train_paths)
val_data = NpyDataset(val_paths)

batch_size = 1
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)

# 训练参数
epochs = 10
best_val_loss = float("inf")
early_stop_counter = 0

# 保存损失值
Loss_list = []
Val_loss_list = []

best_model_path = "EG-SAM/new1/SAM_test/rock_sam_best.pth"

for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    rock_sam.train()

    # 训练阶段
    for data, label in tqdm(trainloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        data, label = data.to(device), label.to(device)

        # 前向传播
        pred = rock_sam(data).squeeze(1)  # 移除多余的通道维度

        # 计算损失
        loss_value = loss(pred, label)

        # 反向传播和优化
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

    train_loss /= len(trainloader)
    Loss_list.append(train_loss)

    # 验证阶段
    rock_sam.eval()
    with torch.no_grad():
        for data_val, label_val in valloader:
            data_val, label_val = data_val.to(device), label_val.to(device)

            # 验证前向传播
            pred_val = rock_sam(data_val).squeeze(1)
            val_loss += loss(pred_val, label_val).item()

    val_loss /= len(valloader)
    Val_loss_list.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(rock_sam.state_dict(), best_model_path)
        print(f"Best model saved at Epoch {epoch + 1}")
    # else:
    #     early_stop_counter += 1
    #     if early_stop_counter >= patience:
    #         print("Early stopping triggered.")
    #         break

    # 额外保存第50轮和第100轮的模型
    if epoch + 1 in [50, 100]:
        save_path = f"EG-SAM/new1/SAM/rock_sam_epoch_{epoch+1}.pth"
        torch.save(rock_sam.state_dict(), save_path)
        print(f"Model saved at Epoch {epoch + 1}")

    # 更新学习率
    scheduler.step(val_loss)


# 保存损失值
# np.save("EG-SAM/new1/SAM/train_loss.npy", np.array(Loss_list))
# np.save("EG-SAM/new1/SAM/val_loss.npy", np.array(Val_loss_list))