import torch 
from torch import nn 
from typing import Sequence

ACTIVATION_FUNC_REGISTRY = {
    "silu": nn.SiLU(inplace = True), 
    "selu": nn.SELU(inplace = True), 
    "relu": nn.ReLU(inplace = True), 
    "leaky_relu": nn.LeakyReLU(inplace = True)
}

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.cfg = cfg
        # Adding input dim and output dim to list of widths
        layer_widths = list(cfg["layer_widths"]) # enforcing list type.
        prev_width = cfg["input_dim"]
        layer_widths.append(cfg["output_dim"])
        activation_func = ACTIVATION_FUNC_REGISTRY[cfg["activation_func"]]

        layers = []
        for width in layer_widths: 
            layers.append(nn.Linear(prev_width, width))
            layers.append(activation_func)
            prev_width = width

        # Removing last activation function if activate_final = False
        if not cfg["activate_final"]: 
            layers = layers[:-1]

        self.model = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.model(x)
    
# class MLP(nn.Module):
#     def __init__(self, cfg):
#         super(MLP, self).__init__()
#         self.cfg = cfg
        
#         layer_widths = list(cfg["layer_widths"])
#         prev_width = cfg["input_dim"]
#         layer_widths.append(cfg["output_dim"])
        
#         activation_func = ACTIVATION_FUNC_REGISTRY[cfg["activation_func"]]

#         layers = []
#         for width in layer_widths: 
#             layers.append(nn.Linear(prev_width, width))
#             layers.append(activation_func)
#             prev_width = width

#         # Removing last activation function if activate_final = False
#         if not cfg["activate_final"]: 
#             layers = layers[:-1]

#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x): 
#         return self.model(x)
    
# class MLP(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int, output_dim: int):
#         super().__init__()

#         layers = []
#         in_dim = input_dim

#         for _ in range(num_hidden_layers):
#             layers.append(nn.Linear(in_dim, hidden_dim))
#             in_dim = hidden_dim

#         layers.append(nn.Linear(in_dim, output_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
        