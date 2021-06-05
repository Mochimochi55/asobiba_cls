import csv
import datetime
import pathlib
import shutil
from dataclasses import InitVar, dataclass
from typing import Any, Dict, List

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


@dataclass
class DataObject:
    """ Objects that manage data.
    Args:
        cfg (Dict): Dataset config
    """
    cfg: Dict
    classes: InitVar = None
    transforms: InitVar = None
    datasets: InitVar = None
    dataloader: InitVar = None

    def __post_init__(self, dummy, dummy2, dummy3, dummy4) -> None:
        if "train" in self.cfg and "val" in self.cfg:
            # Augmentation
            self.transforms = {
                "train": get_transforms(self.cfg["train"]["augmentations"]),
                "val": get_transforms(self.cfg["val"]["augmentations"])
            }
            # Dataset
            self.datasets = self.prepare_datasets()
            # Labels
            self.classes = self.datasets["train"].class_to_idx
            # Dataloader
            self.dataloaders = self.prepare_dataloaders()

        elif "test" in self.cfg:
            # Augmentation
            self.transforms = get_transforms(self.cfg["test"]["augmentations"])
            # Dataset
            self.datasets = ImageFolderForAlbumentations(root=self.cfg["test"]["path"],
                                                         transform=self.transforms)
            # Labels
            self.classes = self.datasets.class_to_idx
            # Dataloader
            self.dataloaders = DataLoader(self.datasets, batch_size=self.cfg["test"]["number_of_sample"],
                                          num_workers=self.cfg["test"]["num_workers"])
        else:
            print("[Error]: Check the config file")

    def prepare_datasets(self) -> Dict:
        """ preparetion datasets.
        Returns:
            Dict: Train dataset and val dataset.
        """
        train_dataset = ImageFolderForAlbumentations(root=self.cfg["train"]["path"],
                                                     transform=self.transforms["train"])
        val_dataset = ImageFolderForAlbumentations(root=self.cfg["val"]["path"],
                                                   transform=self.transforms["val"])
        image_datasets = {"train": train_dataset, "val": val_dataset}

        return image_datasets

    def prepare_dataloaders(self) -> Dict:
        """ preparetion dataloaders.
        Returns:
            Dict: Train dataloader and val dataloader.
        """
        balanced_batch_sampler = BalancedBatchSampler(self.datasets["train"], len(self.classes),
                                                      self.cfg["train"]["number_of_sample"])
        dataloaders = {
            "train": DataLoader(self.datasets["train"], batch_sampler=balanced_batch_sampler,
                                num_workers=self.cfg["train"]["num_workers"]),
            "val": DataLoader(self.datasets["val"], batch_size=self.cfg["train"]["number_of_sample"],
                              num_workers=self.cfg["train"]["num_workers"])
        }

        return dataloaders


@dataclass
class ImgRecoObject:
    """ Image recognition model.
    Args:
        cfg (Dict): Train config
        classes (Dict): class info
        is_train (bool): Train(True) or Test(False), Default is True
    """
    cfg: Dict
    classes: Dict
    is_train: bool = True
    model: InitVar = None
    criterion: InitVar = None
    is_cuda: bool = False
    is_dataparallel: bool = False

    def __post_init__(self, dummy, dummy2) -> None:
        if self.is_train:
            # Model
            self.model = get_model(self.cfg["model"], len(self.classes))
            # Criterion
            self.criterion = eval(self.cfg["criterion"])
        else:
            fname = str(pathlib.Path(self.cfg["wight_path"]).name)
            model_name = fname.split("_")[0]

            # Model
            self.model = get_model(model_name, len(self.classes))
            # Load weight
            self.model.load_state_dict(torch.load(self.cfg["wight_path"]))

        # DataParallel
        if torch.cuda.device_count() > 1:
            self.enable_dataparallel()
            self.is_dataparallel = True

        # Use cuda
        if torch.cuda.is_available():
            self.enable_cuda()
            self.is_cuda = True

    def enable_dataparallel(self) -> None:
        """ Enable dataparallel.
        """
        self.model = torch.nn.DataParallel(self.model)

    def enable_cuda(self) -> None:
        """ Enable cuda.
        """
        self.model.cuda()
        if self.is_train:
            self.criterion.cuda()


@dataclass
class ParamsObject:
    """ Parameters.
    Args:
        cfg (Dict): Train config
    """
    cfg: Dict
    maximum_epoch: InitVar = None
    save_interval: InitVar = None
    optimizer: InitVar = None
    scheduler: InitVar = None

    def __post_init__(self, dummy, dummy2, dummy3, dummy4) -> None:
        # Max epoch
        self.maximum_epoch = self.cfg["maximum_epoch"]

        # Save interval
        self.save_interval = self.cfg["save_interval"]

    def set_optimizer(self, model: Any) -> None:
        """ set optimizer
        Args:
            model (Any): Model object
        """
        # Optimizer
        self.optimizer = eval(self.cfg["optimizer"])

    def set_scheduler(self) -> None:
        """ set scheduler
        """
        # Scheduler
        optimizer = self.optimizer
        self.scheduler = eval(self.cfg["scheduler"])


@dataclass
class LogInfomation:
    """ Manage log.
    Args:
        path (str): Yaml path
        model_name (str): Model name
        classes (Dict): class info
    """
    path: str
    model_name: str
    classes: Dict
    outputs_date: InitVar = None
    outputsdir: InitVar = None
    model_outsputdir: InitVar = None
    tesntorboard_writer: InitVar = None

    def __post_init__(self, dummy, dummy2, dummy3, dummy4) -> None:
        # Datetime
        self.outputs_date = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        # Outputs directory
        self.outputsdir = f"./logs/{self.model_name}_cls{len(self.classes):02}_{self.outputs_date}/"
        pathlib.Path(self.outputsdir).mkdir(exist_ok=True, parents=True)

        # Copy yaml
        fname = pathlib.Path(self.path).name
        shutil.copyfile(self.path, f"{self.outputsdir}{fname}")

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
