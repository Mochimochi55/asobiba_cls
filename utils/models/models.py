from typing import Any

import torch.nn as nn
import torchvision
from efficientnet_pytorch import EfficientNet
from torch.utils import model_zoo


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
        model = EfficientNet.from_pretrained(
            f"efficientnet-{num}", num_classes=classes)

    return model


def get_model_type(name: str) -> str:
    """ Get model type name.

    Args:
        name (str): Model name

    Returns:
        str: Model type
    """

    if "resnet" in name:
        model_type = "resnet"
    elif "EfficientNet" in name:
        model_type = "EfficientNet"

    return model_type
