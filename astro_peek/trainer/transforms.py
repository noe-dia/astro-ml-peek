import torch
import numpy as np 

# CIFAR10 transform -> flatten images (32, 32, 3) -> 3072 and apply patches 
def patch_cifar10(dataset):
    # extract the images from the 2d representations (columns with image and label)
    dataset = dataset.squeeze() # shape (N, 32, 32, 3)
    
    # separate into top and bottom, with shapes top = (N, 16, 32, 3)
    top_half = dataset[:, :16, :, :]
    bottom_half = dataset[:, 16:, :, :]
    print(top_half.shape, bottom_half.shape)
    
    # for each image, extract a random patch from the top half and a random patch from the bottom half. save into arrays
    top_patches = []
    bottom_patches = []
    for image in dataset:
        top_half = image[:16,:,:]
        bottom_half = image[16:,:,:]
        
        random_top_loc = torch.randint(low=0, high=16, size=(1,))
        random_top_patch = top_half[:, random_top_loc:random_top_loc+16, :]
        
        random_bottom_loc = torch.randint(low=0, high=16, size=(1,))
        random_bottom_patch = bottom_half[:, random_bottom_loc:random_bottom_loc+16, :]
        
        top_patches.append(random_top_patch)
        bottom_patches.append(random_bottom_patch)
    
    features = torch.tensor(np.array(top_patches))
    labels = torch.tensor(np.array(bottom_patches))
    
    return features, labels



TRANSFORM_REGISTRY ={
    "cifar10": patch_cifar10 # function to apply the transfrom
}