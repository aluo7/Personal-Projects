# shap_utils.py

import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def compute_shap_values(model, inputs, background=None, layer=None, model_type="vit"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if background is None:
        background = inputs[:5]
    
    if model_type == "3d_resnet":
        inputs = inputs.permute(0, 2, 1, 3, 4)
        background = background.permute(0, 2, 1, 3, 4)
    
    background = background.to(device)
    
    if layer:
        explainer = shap.GradientExplainer((model, layer), background)
    else:
        explainer = shap.GradientExplainer(model, background)
    
    inputs = inputs.to(device)
    shap_values = explainer.shap_values(inputs)
    
    return shap_values

def map_shap_to_image(shap_values, img_size=224, patch_size=16):
    num_patches = img_size // patch_size
    shap_grid = np.abs(shap_values.reshape(num_patches, num_patches, -1)).mean(axis=-1)  # aggregate over emb dim
    
    scale_factor = img_size / num_patches  # bilinear interpolation
    upscaled_shap = scipy.ndimage.zoom(shap_grid, zoom=(scale_factor, scale_factor), order=1)
    
    return upscaled_shap

def visualize_shap(shap_values, original_input, model_type="vit", frame_idx=0, patch_size=16, smooth=False, color_map='jet'):
    if model_type == "vit":
        upscaled_shap = map_shap_to_image(shap_values[0], img_size=224, patch_size=patch_size)  # vit vis using patch mapping
        print(f"SHAP value range before normalization: min={upscaled_shap.min()}, max={upscaled_shap.max()}")
        
        normalized_shap = (upscaled_shap - upscaled_shap.min()) / (upscaled_shap.max() - upscaled_shap.min())
        print(f"Normalized SHAP value range: min={normalized_shap.min()}, max={normalized_shap.max()}")
        
        if smooth:  # gauss smoothing
            normalized_shap = scipy.ndimage.gaussian_filter(normalized_shap, sigma=3)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original_input)
        ax[0].set_title('Original Image')
        ax[0].axis('off')

        ax[1].imshow(original_input)
        im = ax[1].imshow(normalized_shap, cmap=color_map, alpha=0.5)
        ax[1].set_title('Upscaled SHAP Overlay')
        ax[1].axis('off')
        
        fig.colorbar(im, ax=ax[1], orientation='vertical')
        
        plt.tight_layout()
        plt.show()
    
    elif model_type == "3d_resnet":
        frame_shap = shap_values[0][:, frame_idx, :, :]
        original_frame = original_input[0][:, frame_idx, :, :]

        if isinstance(frame_shap, torch.Tensor):
            frame_shap = frame_shap.mean(axis=0).cpu().numpy()
        else:
            frame_shap = frame_shap.mean(axis=0)

        if original_frame.shape[0] > 3:
            original_frame = original_frame[0, :, :]  # select first channel
        else:
            original_frame = original_frame.permute(1, 2, 0).cpu().numpy()

        normalized_shap = (frame_shap - frame_shap.min()) / (frame_shap.max() - frame_shap.min())

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original_frame, cmap="gray" if original_frame.ndim == 2 else None)  # handle grayscale or rgb
        ax[0].set_title('Original Frame')
        ax[0].axis('off')

        ax[1].imshow(original_frame, cmap="gray" if original_frame.ndim == 2 else None)
        if normalized_shap.ndim == 3 and normalized_shap.shape[-1] > 1:
            normalized_shap = normalized_shap.mean(axis=-1)

        im = ax[1].imshow(normalized_shap, cmap=color_map, alpha=0.5)
        ax[1].set_title('SHAP Overlay')
        ax[1].axis('off')
        
        fig.colorbar(im, ax=ax[1], orientation='vertical')
        
        plt.tight_layout()
        plt.show()
