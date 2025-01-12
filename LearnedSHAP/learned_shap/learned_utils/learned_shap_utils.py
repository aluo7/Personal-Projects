# learned_shap_utils.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def compute_shap_from_harsanyi_vit(model, data):
    device = next(model.parameters()).device
    data = data.to(device)
    
    shap_values = []
    
    with torch.no_grad():
        patch_embeddings = model.vit.patch_embed(data)
        transformer_output = model.vit.blocks(patch_embeddings)
        
        num_patches_in_output = transformer_output.size(1)
        if model.patch_contributions.size(0) != num_patches_in_output:
            model.patch_contributions = nn.Parameter(
                torch.randn(num_patches_in_output, model.hidden_dim, device=device),
                requires_grad=True
            )

        if not hasattr(model, 'patch_interaction_contributions') or \
           model.patch_interaction_contributions.size(0) != num_patches_in_output:
            model.patch_interaction_contributions = nn.Parameter(
                torch.randn(num_patches_in_output, num_patches_in_output, model.hidden_dim, device=device),
                requires_grad=True
            )

        contributions_individual = torch.einsum('ph,bph->bp', model.patch_contributions, transformer_output)

        contributions_pairwise = torch.einsum('pph,bph,bqh->bp', model.patch_interaction_contributions, transformer_output, transformer_output)
        
        total_contributions = contributions_individual + contributions_pairwise
        
        for i in range(model.num_patches):
            contribution = total_contributions[:, i]
            shap_values.append(contribution.cpu().detach())
    
    return torch.stack(shap_values, dim=1)

def map_harsanyi_shap_to_image(shap_values, img_size=224, patch_size=16):
    num_patches = img_size // patch_size
    
    shap_grid = np.abs(shap_values.cpu().numpy().reshape(num_patches, num_patches))
    
    scale_factor = img_size / num_patches
    upscaled_shap = scipy.ndimage.zoom(shap_grid, zoom=(scale_factor, scale_factor), order=1)
    
    return upscaled_shap

def visualize_harsanyi_shap(shap_values, original_input, patch_size=16, smooth=False, color_map='jet'):
    upscaled_shap = map_harsanyi_shap_to_image(shap_values, img_size=224, patch_size=patch_size)
    print(f"SHAP value range before normalization: min={upscaled_shap.min()}, max={upscaled_shap.max()}")
    
    normalized_shap = (upscaled_shap - upscaled_shap.min()) / (upscaled_shap.max() - upscaled_shap.min())
    print(f"Normalized SHAP value range: min={normalized_shap.min()}, max={normalized_shap.max()}")
    
    if smooth:
        normalized_shap = scipy.ndimage.gaussian_filter(normalized_shap, sigma=3)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_input)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(original_input)
    im = ax[1].imshow(normalized_shap, cmap=color_map, alpha=0.5)
    ax[1].set_title('SHAP Overlay')
    ax[1].axis('off')
    
    fig.colorbar(im, ax=ax[1], orientation='vertical')
    plt.tight_layout()
    plt.show()

def compute_shap_from_harsanyi_resnet3d(model, data):
    device = next(model.parameters()).device
    data = data.to(device)

    if data.size(1) != 3:
        data = data.permute(0, 2, 1, 3, 4)
    
    shap_values = []

    with torch.no_grad():
        features = model.resnet3d.stem(data)
        features = model.resnet3d.layer1(features)
        features = model.resnet3d.layer2(features)
        features = model.resnet3d.layer3(features)
        features = model.resnet3d.layer4(features)

        batch_size, channels, depth, height, width = features.shape
        flattened_features = features.view(batch_size, channels, -1)
        
        num_cubes_in_output = flattened_features.size(2)
        if model.cube_contributions.size(0) != num_cubes_in_output or model.cube_contributions.size(1) != channels:
            model.cube_contributions = nn.Parameter(
                torch.randn(num_cubes_in_output, channels, device=device),
                requires_grad=True
            )
        
        model.num_cubes = num_cubes_in_output

        contributions = torch.einsum('nc,bcn->bn', model.cube_contributions, flattened_features)

        for i in range(model.num_cubes):
            contribution = contributions[:, i]
            shap_values.append(contribution.cpu().detach())

    return torch.stack(shap_values, dim=1)


def map_3d_resnet_shap_to_image(shap_values, img_size=(2, 7, 7), patch_size=(1, 1, 1)):
    depth_size, img_height, img_width = img_size
    patch_depth, patch_height, patch_width = patch_size
    
    num_depth_patches = depth_size // patch_depth
    num_height_patches = img_height // patch_height
    num_width_patches = img_width // patch_width

    shap_grid = shap_values.cpu().numpy().reshape(
        num_depth_patches, num_height_patches, num_width_patches
    )

    scale_factors = (depth_size / num_depth_patches, img_height / num_height_patches, img_width / num_width_patches)
    upscaled_shap = scipy.ndimage.zoom(shap_grid, zoom=scale_factors, order=1)

    return upscaled_shap


def visualize_3d_resnet_shap(shap_values, original_input, frame_idx=0, patch_size=(1, 1, 1), smooth=True, color_map='jet', img_size=(2, 7, 7)):
    upscaled_shap = map_3d_resnet_shap_to_image(shap_values, img_size=img_size, patch_size=patch_size)
    print(f"SHAP value range before normalization: min={upscaled_shap.min()}, max={upscaled_shap.max()}")
    
    normalized_shap = (upscaled_shap - upscaled_shap.min()) / (upscaled_shap.max() - upscaled_shap.min())
    print(f"Normalized SHAP value range: min={normalized_shap.min()}, max={normalized_shap.max()}")
    
    if smooth:
        normalized_shap = scipy.ndimage.gaussian_filter(normalized_shap, sigma=0.1)  # gauss filter

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    permuted_input = original_input.permute(0, 2, 1, 3, 4)
    original_frame = permuted_input[0, :, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()

    if original_frame.ndim == 2:
        if original_frame.max() <= 1.0:
            original_frame = original_frame * 255
        ax[0].imshow(original_frame.astype(np.uint8), cmap='gray')
    elif original_frame.shape[-1] == 3:
        if original_frame.max() <= 1.0:
            original_frame = (original_frame * 255).astype(np.uint8)
        original_frame = np.clip(original_frame ** 0.8, 0, 255)
        ax[0].imshow(original_frame)
    
    ax[0].set_title('Original Frame')
    ax[0].axis('off')
    
    reshaped_shap = scipy.ndimage.zoom(normalized_shap[frame_idx], 
                                       zoom=(original_frame.shape[0] / normalized_shap.shape[1], 
                                             original_frame.shape[1] / normalized_shap.shape[2]),
                                       order=3)
    
    ax[1].imshow(original_frame)
    im = ax[1].imshow(reshaped_shap, cmap=color_map, alpha=0.6)
    ax[1].set_title('Overlayed Frame')
    ax[1].axis('off')
    
    fig.colorbar(im, ax=ax[1], orientation='vertical')
    plt.tight_layout()
    plt.show()
