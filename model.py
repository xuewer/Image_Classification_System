import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


MODEL_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
}


def build_model(
    num_classes: int,
    pretrained: bool = True,
    model_name: str = "resnet18",
    freeze_backbone: bool = False,
):
    model_name = model_name.lower()
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose from: {', '.join(MODEL_BUILDERS)}"
        )

    builder, weights = MODEL_BUILDERS[model_name]
    model = builder(weights=weights if pretrained else None)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
