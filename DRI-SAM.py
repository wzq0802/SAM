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
from lora import Linear,MergedLinear
from model import UNet
from scipy.ndimage import maximum_filter, label, find_objects
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from torch.nn.utils.rnn import pad_sequence
# 固定随机种子
seed = 82
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

data_path = os.listdir("data/ori")
data_path.sort(key=lambda x: int(x[:-4]))
data_path1 = os.listdir("data/seg")
data_path1.sort(key=lambda x: int(x[:-4]))


train_val_ratio = 0.8
train_size = int(train_val_ratio * len(data_path))

train_paths = data_path[:train_size]
val_paths = data_path[train_size:]


def default_loader(path):
    data_pil = np.load(f"data/ori/{path}").reshape((1, 256, 256))
    data_tensor = torch.tensor(data_pil).type(torch.FloatTensor) / 255.0  


    data_tensor = data_tensor.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
    transform = transforms.Resize((1024, 1024))
    data_tensor = transform(data_tensor)
    return data_tensor


def default_loader1(path):
    data_pil1 = np.load(f"data/seg/{path}").reshape((1, 256, 256))
    data_tensor1 = torch.tensor(data_pil1).type(torch.FloatTensor)
    return data_tensor1


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

class UNetPointGenerator(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.unet = UNet()
        self.unet.load_state_dict(torch.load(model_path))
        self.unet.eval()

    def extract_points(self, image_tensor):

        with torch.no_grad():
            input_tensor = F.interpolate(image_tensor[:, 0:1], size=256, mode='bilinear', align_corners=False) * 255
            prob_map = torch.sigmoid(self.unet(input_tensor))[0, 0].cpu().numpy()

        neighborhood_size = 11
        local_max = maximum_filter(prob_map, size=neighborhood_size) == prob_map
        detected = (prob_map > 0.05) & local_max
        labels, num_features = label(detected)
        slices = find_objects(labels)
        points = []
        for dy, dx in slices:
            x_center = (dx.start + dx.stop - 1) // 2
            y_center = (dy.start + dy.stop - 1) // 2
            points.append([x_center, y_center])
        points = np.array(points)
        return points


def inject_lora_to_sam(sam_model, rank=4):
    for name, module in sam_model.image_encoder.named_children():
        if isinstance(module, nn.ModuleList):  
            for block in module:
               
                block.attn.qkv = MergedLinear(
                    in_features=block.attn.qkv.in_features,
                    out_features=block.attn.qkv.out_features,
                    r=rank,
                    lora_alpha=16,
                    enable_lora=[True, True, True], 
                    merge_weights=False
                )
               
                block.mlp.lin1 = Linear(
                    in_features=block.mlp.lin1.in_features,
                    out_features=block.mlp.lin1.out_features,
                    r=rank,
                    lora_alpha=16,
                    merge_weights=False
                )
                block.mlp.lin2 = Linear(
                    in_features=block.mlp.lin2.in_features,
                    out_features=block.mlp.lin2.out_features,
                    r=rank,
                    lora_alpha=16,
                    merge_weights=False
                )
    return sam_model


class RockSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, point_generator=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.point_generator = point_generator

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):
        B, C, H, W = image.shape  # [B, 3, 1024, 1024]
        image_embeddings = self.image_encoder(image)

        point_coords_batch = []
        point_labels_batch = []
        max_num_points = 0

        for i in range(B):
            img_single = image[i:i+1]  # [1, C, H, W]

            if self.point_generator is not None:
                points = self.point_generator.extract_points(img_single)  # numpy [N, 2]
                if len(points) == 0:
                    print(f"[Warning] No points detected in image {i}, using center fallback")
                    points = np.array([[W // 2, H // 2]])
                points = points * (1024.0 / 256.0)
                coords = torch.tensor(points, dtype=torch.float32, device=image.device)  # [N, 2]
                labels = torch.ones((coords.shape[0],), dtype=torch.float32, device=image.device)  # [N]
            else:
                coords = torch.zeros((0, 2), dtype=torch.float32, device=image.device)
                labels = torch.zeros((0,), dtype=torch.float32, device=image.device)

            point_coords_batch.append(coords)
            point_labels_batch.append(labels)
            max_num_points = max(max_num_points, coords.shape[0])

      
        padded_coords = []
        padded_labels = []
        for coords, labels in zip(point_coords_batch, point_labels_batch):
            pad_len = max_num_points - coords.shape[0]
            if pad_len > 0:
                pad_coords = F.pad(coords, (0, 0, 0, pad_len), value=0)  
                pad_labels = F.pad(labels, (0, pad_len), value=-1)     
            else:
                pad_coords = coords
                pad_labels = labels
            padded_coords.append(pad_coords)
            padded_labels.append(pad_labels)


        point_coords_tensor = torch.stack(padded_coords, dim=0)  # [B, max_N, 2]
        point_labels_tensor = torch.stack(padded_labels, dim=0)  # [B, max_N]

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords_tensor, point_labels_tensor),
            boxes=None,
            masks=None,
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return F.interpolate(
            low_res_masks,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
device = "cuda:1" if torch.cuda.is_available() else "cpu"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
point_generator = UNetPointGenerator("point/pore_binary_unet_epoch.pth").to(device)

sam = inject_lora_to_sam(sam, rank=4)


for name, param in sam.named_parameters():
    if 'lora_A' not in name and 'lora_B' not in name:
        param.requires_grad = False
    else:
        print(f"Trainable layer: {name}") 


for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

for param in sam.mask_decoder.parameters():
    param.requires_grad = True

rock_sam = RockSAM(
    image_encoder=sam.image_encoder,
    mask_decoder=sam.mask_decoder,
    prompt_encoder=sam.prompt_encoder,
    point_generator=point_generator
).to(device)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, rock_sam.parameters()), lr=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.5)

bce_loss = nn.BCEWithLogitsLoss()


def loss(pred, target):
    target_resized = F.interpolate(target, size=pred.shape[1:], mode="bilinear", align_corners=False)
    target_resized = target_resized.squeeze(1)
    loss_bce = bce_loss(pred, target_resized)
    return loss_bce


# 数据加载
train_data = NpyDataset(train_paths)
val_data = NpyDataset(val_paths)

batch_size = 8
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size)


epochs = 100
best_val_loss = float("inf")


Loss_list = []
Val_loss_list = []

best_model_path = "point/SAM_point/rock_sam_best.pth"
best_lora_path = "point/SAM_point/lora_best.pth" 
for epoch in range(epochs):
    train_loss = 0.0
    val_loss = 0.0
    rock_sam.train()

    for data, label_tensor in tqdm(trainloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        data, label_tensor = data.to(device), label_tensor.to(device)

        pred = rock_sam(data).squeeze(1)
        loss_value = loss(pred, label_tensor)

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

    train_loss /= len(trainloader)
    Loss_list.append(train_loss)

    rock_sam.eval()
    with torch.no_grad():
        for data_val, label_val in valloader:
            data_val, label_val = data_val.to(device), label_val.to(device)
            pred_val = rock_sam(data_val).squeeze(1)
            val_loss += loss(pred_val, label_val).item()

    val_loss /= len(valloader)
    Val_loss_list.append(val_loss)

    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    def save_lora_only(model, path):

        lora_state = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_state[name] = param.data
        torch.save(lora_state, path)

    if val_loss < best_val_loss:
        torch.save(rock_sam.state_dict(), best_model_path)
        save_lora_only(sam.image_encoder, best_lora_path)  #


    scheduler.step(val_loss)


np.save("point/SAM_point/train_loss.npy", np.array(Loss_list))
np.save("point/SAM_point/val_loss.npy", np.array(Val_loss_list))

