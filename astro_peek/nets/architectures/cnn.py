from torch import nn
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b2, efficientnet_v2_s, efficientnet_v2_m


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        last_dim = 1280  # For efficientnet_b0
        backbone  = cfg["backbone"]
        if backbone == "efficientnet_b0":
            vision_model = efficientnet_b0()
            self.model = vision_model.features
        elif backbone == "efficientnet_b2":
            vision_model = efficientnet_b2()
            last_dim = 1408
            self.model = vision_model.features
        elif backbone == "efficientnet_v2_s":
            vision_model = efficientnet_v2_s()
            self.model = vision_model.features
        elif backbone == "efficientnet_v2_m":
            vision_model = efficientnet_v2_m()
            self.model = vision_model.features
        else:
            raise ValueError(f"Backbone {backbone} not supported.")
        
        # Reshape head for data
        self.reshape_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=cfg["dropout_rate"], inplace=True),
            )
        
        # Output head
        hidden_dim = cfg["hidden_dim"]
        output_dim = cfg["output_dim"]
        self.head = nn.Sequential(
            nn.Linear(last_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
            )
        
    
    def forward(self, x): 
        # Add channel dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # Repeat channels to match expected input size (only if not using pretrained single-channel adaptation)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        # Compute forward pass x -> f_theta(x)
        features = self.model(x.float())
        features = self.reshape_head(features)
        x = self.head(features)        
        return x