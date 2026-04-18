import torch 
from torch import nn, optim
from astro_peek.utils import load_yaml
from datasets import load_from_disk
from astro_peek.nets.encoder_base import Encoder
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import os 
from .transforms import TRANSFORM_REGISTRY
OPTIMIZER_REGISTRY = {
    "adam": optim.Adam
}


def training(cfg): 

    # Setting the random seed: 
    seed = cfg["trainer"]["seed"]
    torch.manual_seed(seed)

    # Loading dataset
    data_cfg = cfg["data"]
    data_split = data_cfg['data_split'] # (train_size, val_size, test_size) (must be numbers between 0 and 1)
    # dset = load_from_disk(data_cfg["path"])
    # dset = dset.with_format("torch")
    
    # select the train and test set
    dset_train = load_from_disk(data_cfg["train_path"])
    dset_test = load_from_disk(data_cfg["test_path"])
    dset_train = dset_train.with_format("torch")
    test_set = dset_test.with_format("torch")
    
    if data_split is not None:
        dset_train = dset_train.train_test_split(train_size = data_split["train"], test_size=data_split["val"], seed=seed)
        train_set, val_set = dset_train["train"], dset_train["test"]

    # if data_split is not None: 
    #     dset = dset.train_test_split(train_size = data_split["train"], test_size=data_split["test"] + data_split["val"], seed = seed)
    #     train_set, val_w_test_set = dset['train'], dset['test']
    #     val_w_test_set = val_w_test_set.train_test_split(train_size = data_split["val"], test_size=data_split["test"], seed = seed)
    #     val_set, test_set = val_w_test_set['train'], val_w_test_set['test'] 
    else:
        train_set = dset_train
        
    print('training set size: ', train_set.shape)
    print('test set size: ', test_set.shape)
    print('val set size: ', val_set.shape)

    # Instantiating the neural networks: 
    trainer_cfg = cfg["trainer"]
    device = trainer_cfg["device"]
    if device == "auto": 
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device {device}")
    encoder_features_cfg = cfg["encoder_features"]
    encoder_features = Encoder(encoder_features_cfg).to(device)

    encoder_labels_cfg = cfg["encoder_labels"]
    encoder_labels = Encoder(encoder_labels_cfg).to(device)

    assert encoder_labels_cfg["backbone_cfg"]["output_dim"] == encoder_features_cfg["backbone_cfg"]["output_dim"]
    latent_dim =  encoder_labels_cfg["backbone_cfg"]["output_dim"] # output dimension of each net is the latent dim

    # Setting hyperparameters: 
    epochs = trainer_cfg["epochs"]
    batch_size = trainer_cfg["batch_size"]
    optimizer_name = trainer_cfg["optimizer"]
    lr = float(trainer_cfg["lr"])
    optimizer = OPTIMIZER_REGISTRY[optimizer_name](list(encoder_features.parameters()) + list(encoder_labels.parameters()), lr = lr)
    transform_features = trainer_cfg["transform"]

    # val_set = dset["val"].iter()
    losses = []
    epoch_losses = []
    print(epochs)
    for epoch in (pbar:= tqdm(range(int(epochs)))): 
        epoch_loss = 0
        train_loader = train_set.iter(batch_size = batch_size, drop_last_batch=True) # makes the dset an iterable

        for data in train_loader: 
            features, labels = data['image'].to(device), data['theta'].to(device)

            if transform_features is not None:
                features, labels = TRANSFORM_REGISTRY[transform_features](...) # apply transform to get new features 

            optimizer.zero_grad()
            predicted_latent_a = encoder_features(features)
            predicted_latent_b = encoder_labels(labels)
            log_likelihood = (predicted_latent_a * predicted_latent_b).sum(dim = -1) # = log p(y |  x, S) (up to some constant)
            loss = - torch.mean(log_likelihood) # average negative log-likelihood over the batch 
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            epoch_loss += loss.item()
            pbar.set_description(f"Loss = {loss.item():.2g}")   

        epoch_losses.append(epoch_loss)

        # if ((epoch+1)%10) == 0:
        #     fig, axs = plt.subplots(1, 2, figsize = (12, 4))

        #     ax = axs[0]
        #     ax.plot(losses)
        #     ax.set(xlabel = "Optimizer steps", ylabel = "Loss") 

            
        #     ax = axs[1]
        #     ax.plot(epoch_losses)
        #     ax.set(xlabel = "Epochs", ylabel = "Loss") 
        #     plt.show()

    # Saving the models (we might want to change that to save the models during the training loop instead according to some criterion)
    print("Saving encoder features model")
    os.makedirs(encoder_features_cfg["save_dir"], exist_ok = True)
    torch.save(
        {"model": encoder_features.state_dict(), 
        "model_cfg": encoder_features_cfg, 
        "seed": trainer_cfg["seed"]
        }, 
        encoder_features_cfg["save_dir"] + f"seed_{seed}_latentdim_{latent_dim}.pt"
    )
    
    print("Saving encoder labels model")
    os.makedirs(encoder_labels_cfg["save_dir"], exist_ok = True)
    torch.save(
        {"model": encoder_labels.state_dict(), 
        "model_cfg": encoder_labels_cfg, 
        "seed": trainer_cfg["seed"]
        }, 
        encoder_labels_cfg["save_dir"] + f"seed_{seed}_latentdim_{latent_dim}.pt"
    )
    
    
    return (encoder_features, encoder_labels) 