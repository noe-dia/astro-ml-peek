from astro_peek.trainer import training
import hydra 
from omegaconf import DictConfig, OmegaConf
from pprint import pprint

# Hydra decorator.
@hydra.main(
    version_base=None,
    config_path="../configs/", 
    config_name="rings" # the ".yaml" is omitted here.
)
def main(cfg: DictConfig):
    print("Starting training with config:")
    cfg = OmegaConf.to_container(cfg)
    pprint(cfg)

    # Starting training 
    _ = training(cfg) # training func returns the two encoder models (easier to check in a notebook). 




if __name__ == "__main__": 
    main()