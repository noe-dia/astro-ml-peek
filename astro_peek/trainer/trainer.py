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
    data_split = data_cfg['data_split'] # (train_size, val_size, test_size) (must be numbers between 0 and 1)
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
    trainer_cfg = cfg["trainer"]
    device = trainer_cfg["device"]
    encoder_features_cfg = cfg["encoder_features"]
    encoder_features = Encoder(encoder_features_cfg).to(device)

    encoder_labels_cfg = cfg["encoder_labels"]
    encoder_labels = Encoder(encoder_labels_cfg).to(device)

    # Setting hyperparameters: 
    epochs = trainer_cfg["epochs"]
    batch_size = trainer_cfg["batch_size"]
    optimizer_name = trainer_cfg["optimizer"]
    lr = float(trainer_cfg["lr"])
    optimizer = OPTIMIZER_REGISTRY[optimizer_name](list(encoder_features.parameters()) + list(encoder_labels.parameters()), lr = lr)


    train_set = train_set.iter(batch_size = batch_size, drop_last_batch=True)
    # val_set = dset["val"].iter()
    losses = []
    epoch_losses = []
    for epoch in (pbar:= tqdm(range(epochs))): 
        epoch_loss = 0
        for data in train_set: 
            features, labels = data['image'].to(device), data['theta'].to(device)

            optimizer.zero_grad()
            predicted_latent_a = encoder_features(features)
            predicted_latent_b = labels #encoder_labels(labels)
            log_likelihood = (predicted_latent_a * predicted_latent_b).sum(dim = -1) # = log p(y |  x, S) (up to some constant)
            loss = - torch.mean(log_likelihood) # average likelihood over the batch 
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss += loss.item()
            pbar.set_description(f"Epoch = {epoch+1}/{epochs}| Loss = {loss.item()}")   

        epoch_losses.append(epoch_loss)

        if ((epoch+1)%10) == 0:
            fig, axs = plt.subplots(1, 2, figsize = (12, 4))

            ax = axs[0]
            ax.plot(losses)
            ax.set(xlabel = "Optimizer steps", ylabel = "Loss") 

            
            ax = axs[1]
            ax.plot(epoch_losses)
            ax.set(xlabel = "Epochs", ylabel = "Loss") 
            plt.show()
            # plt.savefig("data/")
    
    return (encoder_features, encoder_labels) 