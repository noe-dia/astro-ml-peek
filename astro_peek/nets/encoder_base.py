from torch import nn
from astro_peek.nets import NET_REGISTRY

class Encoder(nn.Module): 
    def __init__(self, cfg):
        """
        backbone = ["cnn", "mlp"]
        backbone_cfg = Dict to instantiate the neural network
        """ 
        backbone = cfg["backbone"]
        backbone_cfg = cfg["backbone_cfg"]
        self.net = NET_REGISTRY[backbone](backbone_cfg)
        
    def forward(self, x): 
        return self.net(x)



    



