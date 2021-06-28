import traceback
from typing import Dict

import numpy as np
import torch

from utils.object import (DataObject, ImgRecoObject, LogInfomation,
                          ParamsObject, SlackNotify)
from utils.pipline import gradcam_run, test_run, train_run
from utils.utils import get_argparse, read_yaml, tracking
import datetime
import pathlib

def train(cfg: Dict) -> None:
    """
    Args:
        cfg (Dict): train config
    """
    # Data instances
    data_obj = DataObject(cfg=cfg["DATASET"])

    # Model instances
    imgreco_obj = ImgRecoObject(
        cfg=cfg["TRAIN_INFO"], classes=data_obj.classes, device=cfg["DEVICE"])

    # Parameter instances
    params_obj = ParamsObject(cfg=cfg["TRAIN_INFO"])
    params_obj.set_optimizer(imgreco_obj.model)
    params_obj.set_scheduler()

    # Log instances
    log_info = LogInfomation(cfg, imgreco_obj.classes)
    for type in ["TRAIN", "VAL"]:
        samples = np.array(data_obj.datasets[type].samples)
        samples = np.delete(samples, 1, 1)
        log_info.dataset_log(type, samples)

    # Informtation
    print("=== Info ===")
    print(f"labels: {data_obj.classes}")
    print(f"model: {cfg['TRAIN_INFO']['MODEL']}")
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
    data_obj = DataObject(cfg=cfg["DATASET"])

    # Model instances
    imgreco_obj = ImgRecoObject(cfg=cfg["TEST_INFO"], classes=data_obj.classes,
                                device=cfg["DEVICE"], is_train=False)

    # Make directory
    now = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    pathlib.Path(f"results/{now}").mkdir(exist_ok=True, parents=True)

    # Informtation
    print("=== Info ===")
    print(f"labels: {data_obj.classes}")
    print(f"wight path: {cfg['TEST_INFO']['WEIGHT_PATH']}")

    # Test start
    test_run(data_obj, imgreco_obj, now)

    if cfg["IS_GRADCAM"]:
        gradcam_run(data_obj, imgreco_obj, now)


def main() -> None:
    # Load settings
    args = get_argparse()
    cfg = read_yaml(args.config)

    # Info
    print(f"torch version: {torch.__version__}")
    print(f"device: {cfg['DEVICE']}\n")

    # Error tracking
    tracker = list()

    # Notify
    if cfg["NOTIFICATION"]["SLACK"]["WEBHOOK_URL"]:
        notify = SlackNotify(cfg["NOTIFICATION"]["SLACK"]["WEBHOOK_URL"],
                             channel=cfg["NOTIFICATION"]["SLACK"]["CHANNEL"],
                             end_message=cfg["NOTIFICATION"]["SLACK"]["END_MESSAGE"])

    if cfg["MODE"] == "train":
        if "MULTIPLE" in cfg:
            for pattern in cfg["MULTIPLE"]:
                try:
                    pattern["DEVICE"] = cfg["DEVICE"]
                    train(pattern)
                except Exception:
                    tracker.append((pattern, traceback.format_exc()))
                    continue
                finally:
                    torch.cuda.empty_cache()
            if tracker:
                tracking(tracker)
        else:
            train(cfg)
    elif cfg["MODE"] == "test":
        test(cfg)
    else:
        print("Set train or test to the mode.")

    # Notice of termination
    if cfg["NOTIFICATION"]["SLACK"]["WEBHOOK_URL"]:
        if notify.end_message:
            notify.end_notify()


if __name__ == "__main__":
    main()
