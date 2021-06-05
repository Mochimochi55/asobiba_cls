from typing import Any, Tuple

import cv2
import numpy as np
from torchvision.datasets.folder import ImageFolder


def img_reader(path: str) -> np.array:
    """ img reader.
    Args:
        path (str): Img path

    Returns:
        np.array: RGB data
    """
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


class ImageFolderForAlbumentations(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=img_reader, is_valid_file=None) -> None:
        super(ImageFolderForAlbumentations, self).__init__(root=root, transform=transform,
                                                           target_transform=target_transform,
                                                           loader=loader, is_valid_file=is_valid_file)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
