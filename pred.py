import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from build_sam import sam_model_registry
from lora import Linear, MergedLinear
from skimage import io, transform
from model import UNet
from scipy.ndimage import maximum_filter, label, find_objects
#
class UNetPointGenerator(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.unet = UNet()
        self.unet.load_state_dict(torch.load(model_path))
        self.unet.eval()

    def extract_points(self, image_tensor, visualize=False):
        with torch.no_grad():
            input_tensor = F.interpolate(image_tensor[:, 0:1], size=256, mode='bilinear', align_corners=False) * 255
            prob_map = torch.sigmoid(self.unet(input_tensor))[0, 0].cpu().numpy()


        threshold = 0.5
        max_points = 50
        yx_coords = np.argwhere(prob_map > threshold)  # shape: [N, 2], order: [y, x]

        if len(yx_coords) == 0:
            print("[Warning] No high-probability points found.")
            return np.array([])

        if len(yx_coords) > max_points:
            selected_idx = np.random.choice(len(yx_coords), max_points, replace=False)
            yx_coords = yx_coords[selected_idx]


        points = yx_coords[:, [1, 0]]

        scale = 1024.0 / 256.0
        points_scaled = points * scale

      
        class UNetPointGenerator(nn.Module):
            def __init__(self, model_path):
                super().__init__()
                self.unet = UNet()
                self.unet.load_state_dict(torch.load(model_path))
                self.unet.eval()

            def extract_points(self, image_tensor):
                with torch.no_grad():
                    input_tensor = F.interpolate(image_tensor[:, 0:1], size=256, mode='bilinear',
                                                 align_corners=False) * 255
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

        return points


def load_model_with_lora(model_type, sam_checkpoint, lora_path, point_generator_path):
    """Load SAM model with LoRA and point generator"""
    # Initialize base SAM model
    sam = sam_model_registry[model_type](checkpoint=None).to(device)

    # Inject LoRA layers
    for name, module in sam.image_encoder.named_children():
        if isinstance(module, nn.ModuleList):
            for block in module:
                block.attn.qkv = MergedLinear(
                    in_features=block.attn.qkv.in_features,
                    out_features=block.attn.qkv.out_features,
                    r=4, lora_alpha=16,
                    enable_lora=[True, True, True],
                    merge_weights=False
                )
                block.mlp.lin1 = Linear(
                    in_features=block.mlp.lin1.in_features,
                    out_features=block.mlp.lin1.out_features,
                    r=4, lora_alpha=16, merge_weights=False
                )
                block.mlp.lin2 = Linear(
                    in_features=block.mlp.lin2.in_features,
                    out_features=block.mlp.lin2.out_features,
                    r=4, lora_alpha=16, merge_weights=False
                )

    point_generator = UNetPointGenerator(point_generator_path).to(device)

    rock_sam = RockSAM(
        image_encoder=sam.image_encoder,
        mask_decoder=sam.mask_decoder,
        prompt_encoder=sam.prompt_encoder,
        point_generator=point_generator
    ).to(device)
    # Load full RockSAM weights (non-strict to handle possible mismatches)
    state_dict = torch.load(sam_checkpoint, map_location=device)
    rock_sam.load_state_dict(state_dict, strict=False)

    # Load LoRA weights separately
    if lora_path:
        lora_weights = torch.load(lora_path, map_location=device)
        for name, param in rock_sam.image_encoder.named_parameters():
            if name in lora_weights:
                param.data.copy_(lora_weights[name])

    return rock_sam


def preprocess_image(image_path):
    """Preprocess image and return tensor with original size"""
    img_np = np.array(Image.open(image_path))
    if img_np.ndim == 2:
        img_np = np.repeat(img_np[:, :, None], 3, axis=-1)
    elif img_np.shape[-1] == 4:
        img_np = img_np[..., :3]

    H, W, _ = img_np.shape
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((1024, 1024), antialias=True)
    ])
    img_tensor = transform(img_np).unsqueeze(0).to(device)
    return img_tensor, (H, W), img_np


class RockSAM(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder, point_generator=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.point_generator = point_generator

    def forward(self, image, point_coords=None, point_labels=None):
        B, C, H, W = image.shape
        image_embeddings = self.image_encoder(image)

        # Handle point prompts
        if point_coords is None and self.point_generator is not None:
            # Auto-generate point prompts
            point_coords_batch = []
            point_labels_batch = []
            max_num_points = 0

            for i in range(B):
                img_single = image[i:i + 1]
                points = self.point_generator.extract_points(img_single)

                if len(points) == 0:
                    points = np.array([[W // 2, H // 2]])  # Fallback to center point

                # Scale points to 1024x1024 size
                points = points * (1024.0 / 256.0)
                coords = torch.tensor(points, dtype=torch.float32, device=device)
                labels = torch.ones((coords.shape[0],), dtype=torch.float32, device=device)

                point_coords_batch.append(coords)
                point_labels_batch.append(labels)
                max_num_points = max(max_num_points, coords.shape[0])

            # Pad to same length
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

            point_coords = torch.stack(padded_coords, dim=0)
            point_labels = torch.stack(padded_labels, dim=0)

        elif point_coords is not None:
            # Use provided point prompts
            assert point_labels is not None, "point_labels must be provided with point_coords"
            point_coords = point_coords.to(device)
            point_labels = point_labels.to(device)
        else:
            # No point prompts
            point_coords = torch.zeros((B, 0, 2), device=device)
            point_labels = torch.zeros((B, 0), device=device)

        # Generate prompt embeddings
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # Predict masks
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


def rocksam_inference(rocksam_model, img_tensor, original_size):
    """Run inference with RockSAM"""
    high_res_masks = rocksam_model(img_tensor)
    binary_segmentation = (
        F.interpolate(high_res_masks, size=original_size, mode="bilinear", align_corners=False)
        .squeeze()
        .detach()
        .cpu()
        .numpy()
    )
    binary_segmentation = (binary_segmentation > 0.5).astype(np.uint8)
    return binary_segmentation


def calculate_metrics(pred_mask, true_mask):
    """Calculate IoU, Dice and Accuracy"""
    pred_mask = pred_mask.astype(np.uint8)
    true_mask = true_mask.astype(np.uint8)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union if union != 0 else 0

    dice = (2.0 * intersection) / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) != 0 else 0

    accuracy = np.sum(pred_mask == true_mask) / (true_mask.shape[0] * true_mask.shape[1])

    return iou, dice, accuracy


if __name__ == "__main__":
    # Path settings

    input_folder = "point/data/test/ori_png"  
    true_mask_folder = "point/data/test/true"  
    output_folder = "point/test"  
    sam_checkpoint = "point/SAM_point/rock_sam_epoch_100.pth"  
    lora_weights = "point/SAM_point/lora_epoch_100.pth"  
    point_generator_path = "point/SAM_point/pore_binary_unet_epoch.pth"  

    os.makedirs(output_folder, exist_ok=True)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize model
    rock_sam = load_model_with_lora(
        "vit_b",
        sam_checkpoint,
        lora_weights,
        point_generator_path
    )
    rock_sam.eval()

    # Calculate metrics
    total_iou, total_dice, total_acc, count = 0, 0, 0, 0

    for file_name in sorted(os.listdir(input_folder)):
        if not file_name.endswith(".png"):
            continue

        input_image_path = os.path.join(input_folder, file_name)
        true_mask_path = os.path.join(true_mask_folder, file_name.replace(".png", ".npy"))
        output_image_path = os.path.join(output_folder, file_name)

        # Preprocess image
        img_tensor, original_size, _ = preprocess_image(input_image_path)

        # Inference (auto-generates point prompts)
        binary_mask = rocksam_inference(
            rock_sam,
            img_tensor,
            (original_size[0], original_size[1])
        )

        # Load ground truth
        true_mask = np.load(true_mask_path).astype(np.uint8)
        if true_mask.shape != binary_mask.shape:
            true_mask = Image.fromarray(true_mask).resize(binary_mask.shape[::-1], Image.NEAREST)
            true_mask = np.array(true_mask)

        # Calculate metrics
        iou, dice, acc = calculate_metrics(binary_mask, true_mask)
        total_iou += iou
        total_dice += dice
        total_acc += acc
        count += 1

        # Save prediction
        Image.fromarray(binary_mask * 255).save(output_image_path)

        print(f"Processed {file_name}: IoU={iou:.4f}, Dice={dice:.4f}, Acc={acc:.4f}")

    # Print average metrics
    avg_iou = total_iou / count if count > 0 else 0
    avg_dice = total_dice / count if count > 0 else 0
    avg_acc = total_acc / count if count > 0 else 0
    print(f"\nAverage Metrics:")
    print(f"IoU: {avg_iou:.4f}")
    print(f"Dice: {avg_dice:.4f}")
    print(f"Accuracy: {avg_acc:.4f}")