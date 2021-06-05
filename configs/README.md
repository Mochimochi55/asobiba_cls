## Train config
List of models :

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

    torch.nn.CrossEntropyLoss()
    torch.nn.BCEWithLogitsLoss()

List of optimizer :

    torch.optim.SGD(model.parameters(), lr=1.0e-4, momentum=0.9)

List of scheduler :

    CosineAnnealingWarmUpRestarts(optimizer, T_0=45, eta_max=1.0e-2, T_up=10)
