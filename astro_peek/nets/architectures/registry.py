from .cnn import CNN
from .mlp import MLP


BACKBONE_REGISTRY = {
    "cnn": CNN, 
    "mlp": MLP
}