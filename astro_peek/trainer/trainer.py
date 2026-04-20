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


def normalize(z):
    return z / np.linalg.norm(z, axis=1, keepdims=True)

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
    test_set = dset['test']

    print('training set size: ', train_set.shape)
    print('test set size: ', test_set.shape)
    # print('val set size: ', val_set.shape)

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

            # Computing similarity scores between all the pairs within the batch for normalization.
            latent_features = normalize(encoder_features(features))
            latent_labels = normalize(encoder_labels(labels))
            logits = latent_features @ latent_labels.T 

            # Positive pairs are on the diagonal
            similarity_score = torch.diag(logits)

            # Normalization is obtained by summing over all the latent labels (the y_0)
            log_normalization = torch.logsumexp(logits, dim=1)
            
            # Log-likelihood = f(x)g(y) - log(normalization)
            log_likelihood = similarity_score - log_normalization
            loss = -log_likelihood.mean() 

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