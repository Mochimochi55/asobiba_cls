import argparse
import csv
import sys
from typing import Dict

import yaml


def get_argparse() -> argparse.Namespace:
    """ Get argsparse.
    Returns:
        argparse.Namespace: Args data (configuration file)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="Please set the path of the configuration file.")
    args = parser.parse_args()

    return args


def read_yaml(path: str) -> Dict:
    """ Read yaml and return the read data.
    Args:
        path (str): Yaml path

    Returns:
        Dict: Yaml data
    """
    try:
        with open(path) as file:
            cfg = yaml.safe_load(file)
    except Exception as e:
        print("Exception occurred while loading yaml...", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    return cfg


def test_logger(data_obj, imgreco_obj, logger) -> None:
    """ Test logging
    Args:
        data_obj (DataObject): DataObject
        imgreco_obj (ImgRecoObject): ImgRecoObject
        logger (Dict): Test result
    """
    with open(logger["log_file"], "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["labels: ", data_obj.classes])
        writer.writerow(["wight path :", imgreco_obj.cfg["wight_path"]])
        writer.writerow(["Total: ", logger["counter"], "Correct: ", logger["correct"],
                         "Miss: ", logger["miss"], "Acc: ", f"{logger['acc']:.2f}%"])
        writer.writerow(["Confusion matrix"])

        for m in logger["matrix"]:
            writer.writerow(m)

        writer.writerow(["Details"])
        writer.writerow(["path", "label", "judge",
                         "pred", "Degree of reliability"])

        for sample, pred, dor in zip(data_obj.datasets.samples, logger["preds_list"], logger["dor_list"]):
            if sample[1] == pred:
                judge = "correct"
            else:
                judge = "miss"

            writer.writerow([sample[0], sample[1], judge, pred, dor])
