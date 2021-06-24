import copy
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

        for phase in ["TRAIN", "VAL"]:
            if phase == "TRAIN":
                imgreco_obj.model.train()
            else:
                imgreco_obj.model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in data_obj.dataloaders[phase]:
                if imgreco_obj.is_cuda:
                    inputs = inputs.to(imgreco_obj.device)
                    labels = labels.to(imgreco_obj.device)
                inputs = inputs.float()
                labels = labels.long()

                # Zero the parameter gradients
                params_obj.optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "TRAIN"):
                    outputs = imgreco_obj.model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = imgreco_obj.criterion(outputs, labels)

                    # Backward + optimize
                    if phase == "TRAIN":
                        params_obj.optimizer.zero_grad()
                        loss.backward()
                        params_obj.optimizer.step()
                        params_obj.scheduler.step()

                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_obj.datasets[phase])
            epoch_acc = running_corrects / len(data_obj.datasets[phase])

            if phase == "TRAIN":
                train_loss = epoch_loss
                log_info.tesntorboard_writer.add_scalar("Loss/Train", epoch_loss, epoch+1)
            else:
                if params_obj.best_params["BESTLOSS"] is None:
                    params_obj.best_params["BESTLOSS"] = epoch_loss

                if epoch_acc > params_obj.best_params["BESTACC"]:
                    params_obj.best_params["BESTACC"] = epoch_acc
                    params_obj.best_params["BESTEP"] = f"{epoch+1:04}"
                    params_obj.best_params["BESTLOSS"] = epoch_loss
                    params_obj.best_params["BESTWEIGHT"] = copy.deepcopy(imgreco_obj.model.state_dict())
                elif epoch_acc == params_obj.best_params["BESTACC"] and epoch_loss < params_obj.best_params["BESTLOSS"]:
                    params_obj.best_params["BESTEP"] = f"{epoch+1:04}"
                    params_obj.best_params["BESTLOSS"] = epoch_loss
                    params_obj.best_params["BESTWEIGHT"] = copy.deepcopy(imgreco_obj.model.state_dict())

                log_info.tesntorboard_writer.add_scalar("Loss/Val", epoch_loss, epoch+1)
                log_info.tesntorboard_writer.add_scalar("Accuracy", epoch_acc, epoch+1)

            # Save
            if epoch == 0 or epoch % params_obj.save_interval == 0:
                model_file_name = f"{imgreco_obj.get_model_name()}_cls{len(log_info.classes):02}_ep{epoch+1:04}.pth"
                model_weight = imgreco_obj.model.state_dict()
                torch.save(model_weight, f"{log_info.model_outsputdir}{model_file_name}")

        elapsed_time = time.time() - start

        max_epoch.set_postfix({"Epoch": epoch+1, "Train_loss": f"{train_loss:.5f}",
                               "VAll_loss": f"{epoch_loss:.5f}", "Val_acc": f"{epoch_acc:.5f}", "Time": f"{elapsed_time}"})

    model_file_name = f"{imgreco_obj.get_model_name()}_BestModel_cls{len(log_info.classes):02}" + \
                      f"_ep{params_obj.best_params['BESTEP']}_acc{params_obj.best_params['BESTACC']:.5f}" + \
                      f"_loss{params_obj.best_params['BESTLOSS']:.5f}.pth"
    torch.save(params_obj.best_params["BESTWEIGHT"], f"{log_info.model_outsputdir}{model_file_name}")


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
                inputs = inputs.to(imgreco_obj.device)
                labels = labels.to(imgreco_obj.device)
            inputs = inputs.float()
            labels = labels.long()

            outputs = imgreco_obj.model(inputs)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            for idx in range(inputs.shape[0]):
                counter += 1

                if labels[idx] == preds[idx]:
                    correct += 1

                if imgreco_obj.is_cuda:
                    labels[idx] = labels[idx].cpu()
                    preds[idx] = preds[idx].cpu()
                    outputs[idx] = outputs[idx].cpu()
                labels_list.append(int(labels[idx]))
                preds_list.append(int(preds[idx]))
                dor_list.append(outputs[idx])

        miss = counter-correct
        acc = (correct / counter) * 100
        matrix = confusion_matrix(labels_list, preds_list)

        logger = {
            "LOG_FILE": log_file,
            "COUNTER": counter,
            "CORRECT": correct,
            "MISS": miss,
            "ACC": acc,
            "MATRIX": matrix,
            "PREDS_LIST": preds_list,
            "DOR_LIST": dor_list}
        test_logger(data_obj, imgreco_obj, logger)

        print( f"\nTotal: {counter}, Correct: {correct}, Miss: {miss}, Acc: {acc:.2f}%")
        print(f"Confusion matrix\n{matrix}")
