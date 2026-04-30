from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, MultiplicativeLR




SCHEDULER_REGISTRY = {
    "exponential_lr": ExponentialLR, 
    "cos_lr": CosineAnnealingLR,
    ""
}