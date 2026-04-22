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

    # Flatten
    features = top_patches.reshape(N, -1)
    labels = bottom_patches.reshape(N, -1)

    return features, labels



TRANSFORM_REGISTRY ={
    "cifar10": patch_cifar10 # function to apply the transfrom
}