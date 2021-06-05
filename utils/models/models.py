from typing import Any

import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet


def get_model(name: str, classes: int) -> Any:
    """ Get model.
    Args:
        name (str): Model name
        classes (int): Number of classes

    Returns:
        Any: Model
    """
    if "resnet" in name:
        model = eval(f"torchvision.models.{name}(pretrained=True)")
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, classes)
    elif "EfficientNet" in name:
        num = name.split("-")[1]
        model = EfficientNet.from_pretrained( f"efficientnet-{num}", num_classes=classes)

    return model
