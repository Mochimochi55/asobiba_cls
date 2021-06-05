from typing import Dict

import torch

from utils.object import DataObject, ImgRecoObject, LogInfomation, ParamsObject
from utils.pipline import train_run, test_run
from utils.utils import get_argparse, read_yaml

import numpy as np
def train(cfg: Dict, path) -> None:
    """
    Args:
        cfg (Dict): train config
        path (str): Yaml path
    """
    # Data instances
    data_obj = DataObject(cfg=cfg["dataset"])

    # Model instances
    imgreco_obj = ImgRecoObject(cfg=cfg["train_info"], classes=data_obj.classes)

    # Parameter instances
    params_obj = ParamsObject(cfg=cfg["train_info"])
    params_obj.set_optimizer(imgreco_obj.model)
    params_obj.set_scheduler()

    # Log instances
    log_info = LogInfomation(path, cfg["train_info"]["model"], imgreco_obj.classes)
    for type in ["train", "val"]:
        samples = np.array(data_obj.datasets[type].samples)
        samples = np.delete(samples, 1, 1)
        log_info.dataset_log(type, samples)

    # Informtation
    print("\n=== Info ===")
    print(f"labels: {data_obj.classes}")
    print(f"model: {cfg['train_info']['model']}")
    print(f"criterion: {imgreco_obj.criterion}")
    print(f"optimizer: {params_obj.optimizer}")
    print(f"scheduler: {params_obj.scheduler}\n")

    # Training start
    train_run(data_obj, imgreco_obj, params_obj, log_info)


def test(cfg) -> None:
    """
    Args:
        cfg (Dict): test config
    """
    # Data instances
    data_obj = DataObject(cfg=cfg["dataset"])

    # Model instances
    imgreco_obj = ImgRecoObject(cfg=cfg["test_info"], classes=data_obj.classes, is_train=False)

    # Informtation
    print("\n=== Info ===")
    print(f"labels: {data_obj.classes}")
    print(f"wight path: {cfg['test_info']['wight_path']}")

    # Test start
    test_run(data_obj, imgreco_obj)


def main() -> None:
    # Info
    print(f"torch version: {torch.__version__}")
    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    # Load settings
    args = get_argparse()
    cfg = read_yaml(args.config)

    if cfg["mode"] == "train":
        train(cfg, args.config)
    elif cfg["mode"] == "test":
        test(cfg)
    else:
        print("Set train or test to the mode.")


if __name__ == "__main__":
    main()
