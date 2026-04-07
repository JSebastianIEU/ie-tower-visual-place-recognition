from torchvision.models import ResNet50_Weights

from src.data.preprocessing import build_default_transform


def get_image_transform(model_name: str = "resnet50"):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V2.transforms()

    return build_default_transform()
