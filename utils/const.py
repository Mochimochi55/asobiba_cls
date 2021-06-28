GRADCAM_LAYER = {
    "RESNET": {
        "IDENTITIES": ["_classifier"],
        "TARGET": "avgpool"
    },
    "EFFICIENTNET": {
        "IDENTITIES": ["_dropout", "_fc", "_swish"],
        "TARGET": "_avg_pooling",
    },
}
