# asobiba_cls

    python version: 3.8.6
    pytorch version: 1.8.1 LTS

Requirements

    pip install -r requirements.txt

datasets

    folder / label_0 / xxx.jpg
                       ...
           / label_1 / xxx.jpg
                       ...

train

    python main.py -c config/train.yaml

test

    python main.py -c config/test.yaml
