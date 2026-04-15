import torch 
from torch import nn, optim
from astro_peek.utils import load_yaml
from datasets import load_from_disk
from astro_peek.nets.encoder_base import Encoder
from tqdm import tqdm 
import matplotlib.pyplot as plt 

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam
}


def training(cfg_dir): 
    cfg = load_yaml(cfg_dir)
    
    # Setting the random seed: 
    seed = cfg["trainer"]["seed"]
    torch.manual_seed(seed)

    # Loading dataset
    data_cfg = cfg["data"]
    data_split = cfg['data_split'] # (train_size, val_size, test_size) (must be numbers between 0 and 1)
    dset = load_from_disk(data_cfg["path"])
    dset = dset.with_format("torch")

    if data_split is not None: 
        dset = dset.train_test_split(train_size = data_split["train"], test_size=data_split["test"] + data_split["val"], seed = seed)
        train_set, val_w_test_set = dset['train'], dset['test']
        val_w_test_set = val_w_test_set.train_test_split(train_size = data_split["val"], test_size=data_split["test"], seed = seed)
        val_set, test_set = val_w_test_set['train'], val_w_test_set['test'] 
    else:
        train_set = dset

    # Instantiating the neural networks: 
    encoder_features_cfg = cfg["encoder_features"]
    encoder_features = Encoder(encoder_features_cfg)

    encoder_labels_cfg = cfg["encoder_labels"]
    encoder_labels = Encoder(encoder_labels_cfg)

    # Setting hyperparameters: 
    trainer_cfg = cfg["trainer"]
    epochs = trainer_cfg["epochs"]
    batch_size = trainer_cfg["batch_size"]
    optimizer_name = trainer_cfg["optimizer"]
    lr = trainer_cfg["lr"]
    optimizer = OPTIMIZER_REGISTRY[optimizer_name](encoder_features.parameters() + encoder_labels.parameters(), lr = lr)
    device = trainer_cfg["device"]


    train_set = dset["train"].iter(batch_size = batch_size, drop_last_batch=True)
    val_set = dset["val"].iter()
    losses = []
    epoch_losses = []
    for epoch in (pbar:= tqdm(epochs)): 
        epoch_loss = 0
        for data in dset: 
            features, labels = data['image'].to(device), data['theta'].to(device)

            optimizer.zero_grad()
            predicted_latent_a = encoder_features(features)
            predicted_latent_b = encoder_labels(labels)
            log_likelihood = (predicted_latent_a * predicted_latent_b).sum(dim = -1) # = log p(y |  x, S) (up to some constant)
            loss = - torch.mean(log_likelihood) # average likelihood over the batch 
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss)
        pbar.set_description(f"Epoch = {epoch}/{epochs}| Loss = {loss.item()}")   

        if ((epoch+1)%10) == 0:
            fig, axs = plt.subplots(1, 2, figsize = (12, 4))

            ax = axs[0]
            ax.plot(losses)
            ax.set(xlabel = "Optimizer steps", ylabel = "Loss") 

            
            ax = axs[1]
            ax.plot(epoch_losses)
            ax.set(xlabel = "Epochs", ylabel = "Loss") 
