import csv
import datetime
import pathlib
from dataclasses import InitVar, dataclass
from typing import Any, Dict, List

import slackweb
import torch
from torch.nn import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.dataset import ImageFolderForAlbumentations
from .data.sampler import BalancedBatchSampler
from .data.transforms import get_transforms
from .models.models import get_model
from .scheduler import CosineAnnealingWarmUpRestarts
from .utils import write_yaml


@dataclass
class DataObject:
    """ Objects that manage data.

    Args:
        cfg (Dict): Dataset in yaml
    """
    cfg: Dict
    classes: InitVar = None
    transforms: InitVar = None
    datasets: InitVar = None
    dataloader: InitVar = None

    def __post_init__(self, dummy, dummy2, dummy3, dummy4) -> None:
        if "TRAIN" in self.cfg and "VAL" in self.cfg:
            # Augmentation
            self.transforms = {
                "TRAIN": get_transforms(self.cfg["TRAIN"]["AUGMENTATIONS"]),
                "VAL": get_transforms(self.cfg["VAL"]["AUGMENTATIONS"])
            }
            # Dataset
            self.datasets = self.prepare_datasets()
            # Labels
            self.classes = self.datasets["TRAIN"].class_to_idx
            # Dataloader
            self.dataloaders = self.prepare_dataloaders()

        elif "TEST" in self.cfg:
            # Augmentation
            self.transforms = get_transforms(self.cfg["TEST"]["AUGMENTATIONS"])
            # Dataset
            self.datasets = ImageFolderForAlbumentations(root=self.cfg["TEST"]["PATH"],
                                                         transform=self.transforms)
            # Labels
            self.classes = self.datasets.class_to_idx
            # Dataloader
            self.dataloaders = DataLoader(self.datasets, batch_size=self.cfg["TEST"]["NUMBER_OF_SAMPLE"],
                                          num_workers=self.cfg["TEST"]["NUM_WORKERS"])
        else:
            print("[Error]: Check the config file. mode is train or test")

    def prepare_datasets(self) -> Dict:
        """ preparetion datasets.

        Returns:
            Dict: Train dataset and val dataset.
        """
        train_dataset = ImageFolderForAlbumentations(root=self.cfg["TRAIN"]["PATH"],
                                                     transform=self.transforms["TRAIN"])
        val_dataset = ImageFolderForAlbumentations(root=self.cfg["VAL"]["PATH"],
                                                   transform=self.transforms["VAL"])
        image_datasets = {"TRAIN": train_dataset, "VAL": val_dataset}

        return image_datasets

    def prepare_dataloaders(self) -> Dict:
        """ preparetion dataloaders.

        Returns:
            Dict: Train dataloader and val dataloader.
        """
        balanced_batch_sampler = BalancedBatchSampler(self.datasets["TRAIN"], len(self.classes),
                                                      self.cfg["TRAIN"]["NUMBER_OF_SAMPLE"])
        dataloaders = {
            "TRAIN": DataLoader(self.datasets["TRAIN"], batch_sampler=balanced_batch_sampler,
                                num_workers=self.cfg["TRAIN"]["NUM_WORKERS"]),
            "VAL": DataLoader(self.datasets["VAL"], batch_size=self.cfg["TRAIN"]["NUMBER_OF_SAMPLE"],
                              num_workers=self.cfg["TRAIN"]["NUM_WORKERS"])
        }

        return dataloaders


@dataclass
class ImgRecoObject:
    """ Image recognition model.

    Args:
        cfg (Dict): Train_info or test_info in yaml
        classes (Dict): Class info
        device (str): Cuda or cpu
        is_train (bool): Train(True) or Test(False), Default is True
        is_cuda (bool): cuda(True) or cpu(False), Default is False
    """
    cfg: Dict
    classes: Dict
    device: str
    is_train: bool = True
    is_cuda: bool = False
    model: InitVar = None
    model_name: str = None
    criterion: InitVar = None

    def __post_init__(self, dummy, dummy2) -> None:
        if self.is_train:
            self.model_name = self.cfg["MODEL"]

            # Model
            self.model = get_model(self.model_name, len(self.classes))
            # Criterion
            self.criterion = eval(self.cfg["CRITERION"])
        else:
            fname = str(pathlib.Path(self.cfg["WEIGHT_PATH"]).name)
            self.model_name = fname.split("_")[0]

            # Model
            self.model = get_model(self.model_name, len(self.classes))
            # Load weight
            self.model.load_state_dict(torch.load(self.cfg["WEIGHT_PATH"]))

        if "cuda" in self.device:
            self.is_cuda = True
            # Set device
            self.set_device()

    def set_device(self) -> None:
        """ Enable cuda.

        """
        self.model.to(self.device)
        if self.is_train:
            self.criterion.to(self.device)

    def get_model_name(self) -> None:
        """ Get model name.

        Returns:
            str: Model name
        """
        return self.model_name


@dataclass
class ParamsObject:
    """ Parameters.

    Args:
        cfg (Dict): Train_config in yaml
    """
    cfg: Dict
    maximum_epoch: InitVar = None
    save_interval: InitVar = None
    optimizer: InitVar = None
    scheduler: InitVar = None
    best_params: InitVar = None

    def __post_init__(self, dummy, dummy2, dummy3, dummy4, dummy5) -> None:
        # Max epoch
        self.maximum_epoch = self.cfg["MAXIMUM_EPOCH"]

        # Save interval
        self.save_interval = self.cfg["SAVE_INTERVAL"]

        # Best parameter
        self.best_params = {"BESTACC": 0, "BESTEP": 0,
                            "BESTLOSS": None, "BESTWEIGHT": None}

    def set_optimizer(self, model: Any) -> None:
        """ Set optimizer.

        Args:
            model (Any): Model object
        """
        # Optimizer
        self.optimizer = eval(self.cfg["OPTIMIZER"])

    def set_scheduler(self) -> None:
        """ Set scheduler.
        """
        # Scheduler
        optimizer = self.optimizer
        self.scheduler = eval(self.cfg["SCHEDULER"])


@dataclass
class LogInfomation:
    """ Manage log.

    Args:
        cfg (Dict): Train config
        classes (Dict): class info
    """
    cfg: Dict
    classes: Dict
    outputs_date: InitVar = None
    outputsdir: InitVar = None
    model_outsputdir: InitVar = None
    tesntorboard_writer: InitVar = None

    def __post_init__(self, dummy, dummy2, dummy3, dummy4) -> None:
        model_name = self.cfg["TRAIN_INFO"]["MODEL"]

        # Datetime
        self.outputs_date = str(
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Outputs directory
        self.outputsdir = f"./logs/{model_name}_cls{len(self.classes):02}_{self.outputs_date}/"
        pathlib.Path(self.outputsdir).mkdir(exist_ok=True, parents=True)

        # Copy yaml
        write_yaml(f"{self.outputsdir}config.yaml", self.cfg)

        # Models directory
        self.model_outsputdir = f"{self.outputsdir}models/"
        pathlib.Path(self.model_outsputdir).mkdir(exist_ok=True, parents=True)

        # TensorBoard
        self.tesntorboard_writer = SummaryWriter(log_dir=self.outputsdir)

    def dataset_log(self, type: str, samples: List) -> None:
        """ Dataset logging.

        Args:
            type (str): Train, Val
            samples (List): Image path list
        """
        with open(f"{self.outputsdir}{type}_datasets.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for sample in samples:
                writer.writerow(sample)


@dataclass
class SlackNotify:
    """ Slack Notification.

    Args:
        webhook_url (str): Webhook URL of Slack
        channel (str): Designated channel
        end_message (str): Messaging to notify at the end of the program
    """
    webhook_url: str
    channel: str = None
    end_message: str = None
    slack: InitVar = None

    def __post_init__(self, dummy) -> None:
        self.slack = slackweb.Slack(url=self.webhook_url)

    def notify(self, text: str) -> None:
        """Notification.

        Args:
            text (str): Message to be notified
        """
        self.slack.notify(text=text, channel=self.channel, user_name=pathlib.Path(__file__).name,
                          icon_emoji=":loudspeaker:")

    def end_notify(self):
        """Notice of Termination.
        """
        self.notify(self.end_message)
