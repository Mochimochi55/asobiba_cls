from typing import Any

from efficientnet_pytorch import EfficientNet


def get_model(name: str, classes: int) -> Any:
    """ Get model.
    Args:
        name (str): Model name
        classes (int): Number of classes

    Returns:
        Any: Model
    """
    if "EfficientNet" in name:
        num = name.split("-")[1]
        model = EfficientNet.from_pretrained( f"efficientnet-{num}", num_classes=classes)

    return model
