from torch import nn
from astro_peek.nets import NET_REGISTRY

class Encoder(nn.Module): 
    def __init__(self, backbone = "", backbone_cfg = {}):
        """
        backbone = ["cnn", "mlp"]
        backbone_cfg = Dict to instantiate the neural network
        """ 

        self.net = NET_REGISTRY[backbone]
        
    def forward(self, x): 
        return self.net(x)



    



