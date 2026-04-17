from torch import nn
from astro_peek.nets.architectures.registry import BACKBONE_REGISTRY
from torch import nn 


class Encoder(nn.Module): 
    def __init__(self, cfg):
        """
        backbone = ["cnn", "mlp"]
        backbone_cfg = Dict to instantiate the neural network
        """ 
        super(Encoder, self).__init__()
        backbone = cfg["backbone"]
        backbone_cfg = cfg["backbone_cfg"]
        self.cfg = cfg
        self.net = BACKBONE_REGISTRY[backbone](backbone_cfg)
        
    def forward(self, x): 
        return self.net(x)



    



