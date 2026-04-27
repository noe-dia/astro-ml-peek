from torch import nn
from astro_peek.nets.architectures.registry import BACKBONE_REGISTRY
import torch 


class Encoder(nn.Module): 
    def __init__(self, cfg = None, checkpoints_dir = None):
        """
        backbone = ["cnn", "mlp"]
        backbone_cfg = Dict to instantiate the neural network
        """ 
        super(Encoder, self).__init__()
        

        if checkpoints_dir is not None: 
            self.load_model(checkpoints_dir)

        else: 
            backbone = cfg["backbone"]
            backbone_cfg = cfg["backbone_cfg"]
            self.cfg = cfg
            self.net = BACKBONE_REGISTRY[backbone](backbone_cfg)
    

    def load_model(self, checkpoints_dir): 
        map_location = "cuda" if torch.cuda.is_available() else "cpu"   
        data_ckpt = torch.load(checkpoints_dir, map_location = map_location)
        cfg = data_ckpt["model_cfg"]
        backbone = cfg["backbone"]
        backbone_cfg = cfg["backbone_cfg"]
        self.cfg = cfg
        self.net = BACKBONE_REGISTRY[backbone](backbone_cfg)
        self.load_state_dict(data_ckpt["model"])
        
    
    def forward(self, x): 
        return self.net(x)



    



