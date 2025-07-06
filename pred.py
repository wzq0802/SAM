
 #
# # #
# import numpy as np
# import os
# from skimage import io, transform
# import torch
# import torch.nn.functional as F
# from build_sam import sam_model_registry
# from EGmodel import RockSAM
#
#
# def rocksam_inference(rocksam_model, img_tensor, H, W):
#    #     high_res_masks, edge = rocksam_model(img_tensor)
#     binary_segmentation = (F.interpolate(high_res_masks, size=(H, W), mode="bilinear", align_corners=False)
#                            .squeeze().cpu().detach().numpy() > 0.5).astype(np.uint8)
#     edge_map = (F.interpolate(edge, size=(H, W), mode="bilinear", align_corners=False)
#                 .squeeze().detach().cpu().numpy())
#     return binary_segmentation, edge_map
#
# def load_model_weights(model, checkpoint_path):
# #     state_dict = torch.load(checkpoint_path, map_location="cpu")
#     new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
#     model.load_state_dict(new_state_dict)
#     return model
#
# if __name__ == "__main__":
#     
#     input_folder = "ori_png"
#     output_seg_folder = "pred"
#     output_edge_folder = "edge"
#     model_checkpoint_path = "rock_sam_epoch_100.pth"
#
#     
#     os.makedirs(output_seg_folder, exist_ok=True)
#     os.makedirs(output_edge_folder, exist_ok=True)
#
#  
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
#    
#     sam_model = sam_model_registry["vit_b"](checkpoint=None)
#     rocksam_model = RockSAM(
#         image_encoder=sam_model.image_encoder,
#         mask_decoder=sam_model.mask_decoder,
#         prompt_encoder=sam_model.prompt_encoder,
#     ).to(device)
#
#   
#     rocksam_model = load_model_weights(rocksam_model, model_checkpoint_path)
#     rocksam_model.eval()

#     image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")],
#                          key=lambda x: int(os.path.splitext(x)[0]))  
#     for img_name in image_files:
#         input_path = os.path.join(input_folder, img_name)
#         output_seg_path = os.path.join(output_seg_folder, img_name)
#         output_edge_path = os.path.join(output_edge_folder, img_name)
#
#     
#         img_np = io.imread(input_path)
#         if len(img_np.shape) == 2:
#             img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
#         elif img_np.shape[-1] == 4:
#             img_3c = img_np[..., :3]
#         else:
#             img_3c = img_np
#
#         H, W, _ = img_3c.shape
#         img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
#         img_1024 = img_1024 / 255.0
#
#         img_1024_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
#
#      
#         binary_segmentation, edge_map = rocksam_inference(rocksam_model, img_1024_tensor, H, W)
#
#     
#         io.imsave(output_seg_path, (binary_segmentation * 255).astype(np.uint8))
#         io.imsave(output_edge_path, (edge_map * 255).astype(np.uint8))
#         print(f"Processed: {img_name}")
# # #





import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from build_sam import sam_model_registry
from EGmodel import RockSAM
from sklearn.decomposition import PCA

def show_embedding_slice(embedding, slice_index=0):

    plt.figure(figsize=(5, 5))
    plt.imshow(embedding[0, slice_index].cpu().numpy(), cmap='viridis')
    plt.title(f"Channel {slice_index}")
    plt.axis("off")
    plt.show()


def show_all_embedding_slices(embedding, n_cols=8, max_channels=32):

    total_channels = min(embedding.shape[1], max_channels)
    n_rows = int(np.ceil(total_channels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()
    for i in range(total_channels):
        axes[i].imshow(embedding[0, i].cpu().numpy(), cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f"{i}", fontsize=8)
    for j in range(total_channels, len(axes)):
        axes[j].axis("off")
    plt.suptitle("Embedding Channels")
    plt.tight_layout()
    plt.show()


def show_pca_rgb_embedding(embedding):

    B, C, H, W = embedding.shape
    emb_reshaped = embedding[0].permute(1, 2, 0).reshape(-1, C).cpu().numpy()
    pca = PCA(n_components=3)
    emb_pca = pca.fit_transform(emb_reshaped)
    emb_pca = (emb_pca - emb_pca.min()) / (emb_pca.max() - emb_pca.min())
    emb_rgb = emb_pca.reshape(H, W, 3)
    plt.figure(figsize=(6, 6))
    plt.imshow(emb_rgb)
    plt.axis('off')
    plt.title("PCA RGB Embedding")
    plt.show()


def load_model_weights(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)
    return model


@torch.no_grad()
def extract_fused_embeddings(img_path, checkpoint_path, model_type="vit_b", device=None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")


    img_np = io.imread(img_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    elif img_np.shape[-1] == 4:
        img_3c = img_np[:, :, :3]
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape


    img_1024 = transform.resize(img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_1024).permute(2, 0, 1).unsqueeze(0).float().to(device)

    sam_model = sam_model_registry[model_type](checkpoint=None)
    rocksam_model = RockSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    rocksam_model = load_model_weights(rocksam_model, checkpoint_path)
    rocksam_model.eval()

    image_embeddings = rocksam_model.image_encoder(img_tensor)
    edge = rocksam_model.edge_head(image_embeddings)
    fused_embeddings = rocksam_model.edge_fusion(image_embeddings, edge)
    return fused_embeddings

if __name__ == "__main__":
    image_path = "data/test/ori_png/7.png"
    model_path = "rock_sam_epoch_100.pth"

    fused_embeddings = extract_fused_embeddings(image_path, model_path)

    show_embedding_slice(fused_embeddings, slice_index=0)
    show_all_embedding_slices(fused_embeddings, max_channels=32)
    show_pca_rgb_embedding(fused_embeddings)
