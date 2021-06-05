## Train config
List of models :

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152
    EfficientNet-b0
    EfficientNet-b1
    EfficientNet-b2
    EfficientNet-b3
    EfficientNet-b4
    EfficientNet-b5
    EfficientNet-b6
    EfficientNet-b7
    EfficientNet-b8
    EfficientNet-l2

List of criterion :

    CrossEntropyLoss()
    BCEWithLogitsLoss()

List of optimizer :

    SGD(model.parameters(), lr=1.0e-4, momentum=0.9)
    Adagrad(model.parameters(), lr=1.0e-4)
    Adam(model.parameters(), lr=1.0e-4)
    Adamax(model.parameters(), lr=1.0e-4)

List of scheduler :

    StepLR(optimizer, step_size=20, gamma=0.5)
    ExponentialLR(optimizer, gamma=0.95)
    CosineAnnealingLR(optimizer, T_max=20, eta_min=1.0e-2)
    CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1.0e-2)
