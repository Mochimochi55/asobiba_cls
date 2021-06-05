from typing import Dict

from albumentations import *
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(aug_dict: Dict) -> Compose:
    """ Read augmentation information.
    Args:
        aug_dict (Dict): Dictionary of classes and parameters

    Returns:
        Compose: Augmentation list by albamentations
    """
    compose = list()

    for class_obj, params_obj in aug_dict.items():
        compose.append(eval(class_obj+params_obj))
    compose.append(ToTensorV2())

    transforms = Compose(compose)

    return transforms
