import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class GlobalFeatureExtractor(nn.Module):
    """Wrap a CNN backbone and expose a flattened embedding vector."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return torch.flatten(features, start_dim=1)


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_feature_extractor(
    model_name: str = "resnet50",
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    model_name = model_name.lower()

    if model_name != "resnet50":
        raise ValueError(
            f"Unsupported model '{model_name}'. Available options: ['resnet50']"
        )

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = resnet50(weights=weights)
    embedding_dim = backbone.fc.in_features
    feature_layers = list(backbone.children())[:-1]

    model = GlobalFeatureExtractor(nn.Sequential(*feature_layers))
    model.eval()
    return model, embedding_dim
