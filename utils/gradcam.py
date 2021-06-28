# Clean
from dataclasses import InitVar, dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class GradCAM(object):
    model: Any
    target_layer: Any
    device: str
    is_cuda: bool = False
    feature_map: InitVar = 0
    grad: InitVar = 0

    def __post_init__(self, dummy, dummy2) -> None:
        if "cuda" in self.device:
            self.is_cuda = True
            self.set_device()

        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                module[1].register_forward_hook(self.save_feature_map)
                module[1].register_backward_hook(self.save_grad)

    def set_device(self) -> None:
        """ Enable cuda.

        """
        self.model.to(self.device)

    def save_feature_map(self, module, input, output):
        self.feature_map = output[0].detach()

    def save_grad(self, module, grad_in, grad_out):
        self.grad = grad_out[0].detach()

    def __call__(self, x, index=None):
        x = x.clone()
        if self.is_cuda:
            x = x.to(self.device)

        self.model.zero_grad()

        output = self.model(x)

        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad_()

        if self.is_cuda:
            one_hot = torch.sum(one_hot.to(self.device) * output)
        else:
            one_hot = torch.sum(one_hot * output)
        one_hot.backward()

        self.feature_map = self.feature_map.cpu().data.numpy()
        self.weights = np.mean(self.grad.cpu().data.numpy()[0], axis=(1, 2))
        cam = np.sum(self.feature_map.T * self.weights, axis=2).T
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.size()[-1], x.size()[-2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, index


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


@dataclass
class GuidedBackProp():
    model: Any
    device: str
    is_cuda: bool = False

    def __post_init__(self) -> None:
        if "cuda" in self.device:
            self.is_cuda = True
            self.set_device()

        for module in self.model.named_modules():
            module[1].register_backward_hook(self.bp_relu)

    def set_device(self) -> None:
        """ Enable cuda.

        """
        self.model.to(self.device)

    def bp_relu(self, module, grad_in, grad_out):
        if isinstance(module, nn.ReLU):
            return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, x, index=None):
        x = x.clone()
        if self.is_cuda:
            x = x.to(self.device)

        x.requires_grad_()

        output = self.model(x)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad_()

        if self.is_cuda:
            one_hot = torch.sum(one_hot.to(self.device) * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward()

        if self.is_cuda:
            x = x.grad.cpu()
        result = x.numpy()[0]
        result = np.transpose(result, (1, 2, 0))
        return result, index


def arrange_img(img):
    img = np.maximum(img, 0)
    res = img - img.min()
    res /= res.max()
    res = np.uint8(res * 255)

    return res
