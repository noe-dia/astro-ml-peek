import torch 
from torch import nn, optim
from astro_peek.utils import load_yaml
from datasets import load_from_disk
from astro_peek.nets.encoder_base import Encoder
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import os 
from .transforms import TRANSFORM_REGISTRY
from torch_ema import ExponentialMovingAverage
from info_nce import InfoNCE, info_nce

OPTIMIZER_REGISTRY = {
    "adam": optim.Adam
}


def normalize(z):
    return z / torch.linalg.norm(z, dim=1, keepdim=True)

def training(cfg): 

    # Setting the random seed: 
    seed = cfg["trainer"]["seed"]
    torch.manual_seed(seed)

    # Loading dataset
    data_cfg = cfg["data"]
    data_split = data_cfg['data_split'] # (train_size, val_size, test_size) (must be numbers between 0 and 1)

    # The func load_from_disk will separate the train and the test folder of the directory on its own. 
    dset = load_from_disk(data_cfg["path"])
    dset = dset.with_format("torch")
    
    train_set = dset['train']
    val_set = dset['test']

    print('training set size: ', train_set.shape)
    print('val set size: ', val_set.shape)
    # print('val set size: ', val_set.shape)

    # Instantiating the neural networks: 
    trainer_cfg = cfg["trainer"]
    device = trainer_cfg["device"]
    if device == "auto": 
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device {device}")
    encoder_features_cfg = cfg["encoder_features"]
    encoder_features = Encoder(cfg = encoder_features_cfg).to(device)

    encoder_labels_cfg = cfg["encoder_labels"]
    encoder_labels = Encoder(cfg = encoder_labels_cfg).to(device)

    assert encoder_labels_cfg["backbone_cfg"]["output_dim"] == encoder_features_cfg["backbone_cfg"]["output_dim"]
    latent_dim =  encoder_labels_cfg["backbone_cfg"]["output_dim"] # output dimension of each net is the latent dim

    # Setting hyperparameters: 
    epochs = trainer_cfg["epochs"]
    batch_size = trainer_cfg["batch_size"]
    optimizer_name = trainer_cfg["optimizer"]
    lr = float(trainer_cfg["lr"])
    optimizer = OPTIMIZER_REGISTRY[optimizer_name](list(encoder_features.parameters()) + list(encoder_labels.parameters()), lr = lr)
    ema = ExponentialMovingAverage(list(encoder_features.parameters()) + list(encoder_labels.parameters()), decay=0.995)
    loss_fn= InfoNCE()

    transform_features = trainer_cfg["transform"]

    # val_set = dset["val"].iter()
    losses = []
    epoch_losses = []
    val_losses = []
    val_loss = 0
    print(epochs)
    for epoch in (pbar:= tqdm(range(int(epochs)))): 
        epoch_loss = 0
        train_loader = train_set.iter(batch_size = batch_size, drop_last_batch=True) # makes the dset an iterable
        val_loader = val_set.iter(batch_size = batch_size, drop_last_batch=True)
        for data in train_loader: 
            features, labels = data['image'].to(device), data['theta'].to(device)
            
            if transform_features is not None:
                features, labels = TRANSFORM_REGISTRY[transform_features](...) # apply transform to get new features 
            if labels.ndim == 1: 
                labels = labels.unsqueeze(1)
            optimizer.zero_grad()

            # Computing similarity scores between all the pairs within the batch for normalization.
            latent_features = encoder_features(features)
            latent_labels = encoder_labels(labels)
            # S = torch.tensor([[0.0], [1.0]]).to(device) # just for testing
            # latent_S = encoder_labels(S)
            if trainer_cfg["normalize"]:
                latent_labels = normalize(latent_labels)
                latent_features = normalize(latent_features) 
                # latent_S = normalize(latent_S)


            # logits = latent_features @ latent_labels.T 
            # log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            # loss = -torch.diag(log_probs).mean()
            # if abs(loss - trainer_cfg["loss_criterion"]) < float(trainer_cfg["delta_criterion"]):
            #     break
            loss = loss_fn(latent_features, latent_labels)
                                
            
            loss.backward()
            optimizer.step()
            ema.update()
            losses.append(loss.item())
            epoch_loss += loss.item()
            pbar.set_description(f"Train Loss = {loss.item():.3f} | Val Loss = {val_loss:.3f}| Epoch train loss = {epoch_loss:.3f} | ")   
        epoch_losses.append(epoch_loss)
        # if abs(loss - trainer_cfg["loss_criterion"]) < float(trainer_cfg["delta_criterion"]):
        #     print(f"Loss criterion reached; Current loss: {loss}, Criterion loss: {trainer_cfg['loss_criterion']}")
        #     break

        with torch.no_grad(): 
            for data in val_loader: 
                features, labels = data['image'].to(device), data['theta'].to(device)
            
                if transform_features is not None:
                    features, labels = TRANSFORM_REGISTRY[transform_features](...) # apply transform to get new features 
                if labels.ndim == 1: 
                    labels = labels.unsqueeze(1)

                # Computing similarity scores between all the pairs within the batch for normalization.
                latent_features = encoder_features(features)
                latent_labels = encoder_labels(labels)
                if trainer_cfg["normalize"]:
                    latent_labels = normalize(latent_labels)
                    latent_features = normalize(latent_features) 

                # logits = latent_features @ latent_labels.T 
                # log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
                # val_loss = -torch.diag(log_probs).mean()
                val_loss = loss_fn(latent_features, latent_labels)
                val_losses.append(val_loss.item())

        


        if ((epoch+1)%10) == 0:
            fig, axs = plt.subplots(1, 2, figsize = (12, 4))

            ax = axs[0]
            ax.plot(losses, label = "Training")
            ax.plot(val_losses, label = "Validation")
            ax.set(xlabel = "Optimizer steps", ylabel = "Loss") 
            ax.legend()

            
            ax = axs[1]
            ax.plot(epoch_losses, label = "Training")
            ax.set(xlabel = "Epochs", ylabel = "Loss") 
            ax.legend()
            plt.show()

            plt.savefig(encoder_features_cfg["save_dir"] + f"loss_seed_{seed}_latentdim_{latent_dim}.pdf", bbox_inches = "tight")

    # Saving the models (we might want to change that to save the models during the training loop instead according to some criterion)
    print("Saving encoder features model")
    os.makedirs(encoder_features_cfg["save_dir"], exist_ok = True)
    torch.save(
        {"model": encoder_features.state_dict(), 
        "model_cfg": encoder_features_cfg, 
        "seed": trainer_cfg["seed"],
         "ema": ema.state_dict()
        }, 
        encoder_features_cfg["save_dir"] + f"seed_{seed}_latentdim_{latent_dim}.pt"
    )
    
    print("Saving encoder labels model")
    os.makedirs(encoder_labels_cfg["save_dir"], exist_ok = True)
    torch.save(
        {"model": encoder_labels.state_dict(), 
        "model_cfg": encoder_labels_cfg, 
        "seed": trainer_cfg["seed"], 
         "ema": ema.state_dict()
        }, 
        encoder_labels_cfg["save_dir"] + f"seed_{seed}_latentdim_{latent_dim}.pt"
    )
    
    
    return (encoder_features, encoder_labels, losses, epoch_losses, val_losses) 