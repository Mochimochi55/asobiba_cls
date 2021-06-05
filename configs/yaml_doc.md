
model:
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

criterion:
    torch.nn.CrossEntropyLoss()
    torch.nn.BCEWithLogitsLoss()

optimizer:
    torch.optim.SGD(model.parameters(), lr=1.0e-4, momentum=0.9)

scheduler:
    CosineAnnealingWarmUpRestarts(optimizer, T_0=45, eta_max=1.0e-2, T_up=10)
