import torch
import numpy as np 
# CIFAR10 transform -> flatten images (32, 32, 3) -> 3072 and apply patches 
# def patch_cifar10(dataset):
#     # extract the images from the 2d representations (columns with image and label)
#     dataset = dataset.squeeze()
#     N = len(dataset)
    
#     # for each image, extract a random patch from the top half and a random patch from the bottom half. save into arrays
#     top_patches = []
#     bottom_patches = []

#     for image in dataset:
#         top_half = image[:16,:,:]
#         bottom_half = image[16:,:,:]
        
#         random_top_loc = torch.randint(low=0, high=16, size=(1,))
#         random_top_patch = top_half[:, random_top_loc:random_top_loc+16, :]
        
#         random_bottom_loc = torch.randint(low=0, high=16, size=(1,))
#         random_bottom_patch = bottom_half[:, random_bottom_loc:random_bottom_loc+16, :]
        
#         top_patches.append(random_top_patch)
#         bottom_patches.append(random_bottom_patch)

#     features = torch.stack(top_patches).reshape(N, -1)
#     labels = torch.stack(bottom_patches).reshape(N, -1)
    
    
#     return features, labels

def patch_cifar10(dataset):
    # dataset: (N, C, 32, 32)
    dataset = dataset.squeeze()
    N, H, W, C = dataset.shape

    # Split into top / bottom halves along height
    top_half = dataset[:, :16, :, :]     # (N, 16, 32, C)
    bottom_half = dataset[:, 16:, :, :]  # (N, 16, 32, C)

    # Random horizontal start indices (per image)
    top_idx = torch.randint(0, 17, (N,), device=dataset.device)
    bot_idx = torch.randint(0, 17, (N,), device=dataset.device)

    # Create column indices for 16-wide patches
    arange_w = torch.arange(16, device=dataset.device)  # (16,)

    # Shape: (N, 16)
    top_cols = top_idx[:, None] + arange_w[None, :]
    bot_cols = bot_idx[:, None] + arange_w[None, :]

    # Expand to match dimensions for gather
    # target shape: (N, 16, 16, C)
    top_cols = top_cols[:, None, :, None].expand(N, 16, 16, C)
    bot_cols = bot_cols[:, None, :, None].expand(N, 16, 16, C)

    # Gather along width dimension (dim=2)
    top_patches = torch.gather(top_half, dim=2, index=top_cols)
    bottom_patches = torch.gather(bottom_half, dim=2, index=bot_cols)

    return top_patches, bottom_patches

def augment_patches(
    top_patches: torch.Tensor,
    bottom_patches: torch.Tensor,
    n_color_drop: int = 1,
    n_brightness: int = 1,
    brightness_delta: float = 0.25,
    include_original: bool = True,
    one_aug_per_image: bool = True,
):
    """
    If one_aug_per_image=True:
      - output size stays N
      - for each image, randomly choose ONE augmentation type:
        brightness OR color-drop

    If one_aug_per_image=False:
      - output size is N * (include_original + n_color_drop + n_brightness)
    """

    def _check(x, name):
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"{name} must have shape (N,H,W,3), got {tuple(x.shape)}")

    def _color_drop_with_noise(x):
        N, H, W, _ = x.shape
        keep_idx = torch.randint(0, 3, (N,), device=x.device)
        y = x.clone()
        row_idx = torch.arange(N, device=x.device)

        kept = y[row_idx, :, :, keep_idx]  # (N,H,W)
        kept_std = kept.reshape(N, -1).std(dim=1, unbiased=False).clamp_min(1e-12)
        noise_std = kept_std / 100.0

        for c in range(3):
            mask = keep_idx != c
            if mask.any():
                n = mask.sum()
                noise = torch.randn((n, H, W), device=x.device, dtype=x.dtype) * noise_std[mask][:, None, None]
                y[mask, :, :, c] = noise
        return y

    def _brightness_jitter(x):
        N = x.shape[0]
        delta = (torch.rand(N, device=x.device, dtype=x.dtype) * 2 - 1) * brightness_delta
        y = x + delta[:, None, None, None]
        return y.clamp(0.0, 1.0)

    def _one_aug_mode(x):
        N = x.shape[0]
        y_cd = _color_drop_with_noise(x)
        y_br = _brightness_jitter(x)

        choose_brightness = torch.rand(N, device=x.device) < 0.5
        out = y_cd.clone()
        out[choose_brightness] = y_br[choose_brightness]
        return out

    def _multi_aug_mode(x):
        outs = []
        if include_original:
            outs.append(x)
        for _ in range(n_color_drop):
            outs.append(_color_drop_with_noise(x))
        for _ in range(n_brightness):
            outs.append(_brightness_jitter(x))
        return torch.cat(outs, dim=0)

    _check(top_patches, "top_patches")
    _check(bottom_patches, "bottom_patches")

    if one_aug_per_image:
        aug_top = _one_aug_mode(top_patches)
        aug_bottom = _one_aug_mode(bottom_patches)
    else:
        aug_top = _multi_aug_mode(top_patches)
        aug_bottom = _multi_aug_mode(bottom_patches)
    
    # reshape to make them 1D for training the MLP 
    N = aug_top.shape[0]
    features = aug_top.reshape(N, -1)
    labels = aug_bottom.reshape(N, -1)
    
    return features, labels



TRANSFORM_REGISTRY ={
    "cifar10": patch_cifar10 # function to apply the transfrom
}