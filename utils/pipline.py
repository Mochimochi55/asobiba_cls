import datetime
import pathlib
import time

import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .object import DataObject, ImgRecoObject, LogInfomation, ParamsObject
from .utils import test_logger


def train_run(data_obj: DataObject, imgreco_obj: ImgRecoObject,
              params_obj: ParamsObject, log_info: LogInfomation) -> None:
    """ Start Training.

    Args:
        data_obj (DataObject): DataObject
        imgreco_obj (ImgRecoObject): ImgRecoObject
        params_obj (ParamsObject): ParamsObject
        log_info (LogInfomation): LogInfomation
    """
    max_epoch = tqdm(range(params_obj.maximum_epoch))

    for epoch in max_epoch:
        start = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                imgreco_obj.model.train()
            else:
                imgreco_obj.model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_obj.dataloaders[phase]:
                if imgreco_obj.is_cuda:
                    inputs = inputs.cuda().float()
                    labels = labels.cuda().long()

                # Zero the parameter gradients
                params_obj.optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = imgreco_obj.model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = imgreco_obj.criterion(outputs, labels)

                    # Backward + optimize
                    if phase == "train":
                        params_obj.optimizer.zero_grad()
                        loss.backward()
                        params_obj.optimizer.step()
                        params_obj.scheduler.step()

                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_obj.datasets[phase])
            epoch_acc = running_corrects / len(data_obj.datasets[phase])

            if phase == "train":
                train_loss = epoch_loss
                log_info.tesntorboard_writer.add_scalar(
                    "Loss/Train", epoch_loss, epoch+1)
            else:
                log_info.tesntorboard_writer.add_scalar(
                    "Loss/Val", epoch_loss, epoch+1)
                log_info.tesntorboard_writer.add_scalar(
                    "Accuracy", epoch_acc, epoch+1)

            # Save
            if epoch == 0 or epoch % params_obj.save_interval == 0:
                model_file_name = f"{imgreco_obj.get_model_name()}_cls{len(log_info.classes):02}_ep{epoch+1:04}.pth"
                if imgreco_obj.is_dataparallel:
                    model_weight = imgreco_obj.model.module.state_dict()
                else:
                    model_weight = imgreco_obj.model.state_dict()
                torch.save(
                    model_weight, f"{log_info.model_outsputdir}{model_file_name}")

        elapsed_time = time.time() - start

        max_epoch.set_postfix({"Epoch": epoch+1, "Train_loss": f"{train_loss:.5f}",
                               "VAll_loss": f"{epoch_loss:.5f}", "Val_acc": f"{epoch_acc:.5f}", "Time": f"{elapsed_time}"})


def test_run(data_obj: DataObject, imgreco_obj: ImgRecoObject) -> None:
    """ Start Testing.

    Args:
        data_obj (DataObject): DataObject
        imgreco_obj (ImgRecoObject): ImgRecoObject
    """
    # Log
    log_file = f"results/{str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))}.csv"
    pathlib.Path("results").mkdir(exist_ok=True, parents=True)

    imgreco_obj.model.eval()

    counter = 0
    correct = 0

    labels_list = list()
    preds_list = list()
    dor_list = list()

    with torch.no_grad():
        for inputs, labels in data_obj.dataloaders:
            if imgreco_obj.is_cuda:
                inputs = inputs.cuda().float()
                labels = labels.cuda().float()

            outputs = imgreco_obj.model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            for idx in range(inputs.shape[0]):
                counter += 1

                if labels[idx] == preds[idx]:
                    correct += 1

                labels_list.append(int(labels[idx].cpu()))
                preds_list.append(int(preds[idx].cpu()))
                dor_list.append(outputs[idx].cpu())

        miss = counter-correct
        acc = (correct / counter) * 100
        matrix = confusion_matrix(labels_list, preds_list)

        logger = {
            "log_file": log_file,
            "counter": counter,
            "correct": correct,
            "miss": miss,
            "acc": acc,
            "matrix": matrix,
            "preds_list": preds_list,
            "dor_list": dor_list}
        test_logger(data_obj, imgreco_obj, logger)

        print(
            f"\nTotal: {counter}, Correct: {correct}, Miss: {miss}, Acc: {acc:.2f}%")
        print(f"Confusion matrix\n{matrix}")
