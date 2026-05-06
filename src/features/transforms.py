"""Per-backbone image preprocessing.

DINOv2 expects the same ImageNet-style normalisation as ResNet50, but
the input must be sized so that the side length is a multiple of the
patch size (14). We use 224x224 for speed; 518x518 would give slightly
better features at much higher cost.
"""
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, build_default_transform


def _dinov2_transform(image_size: int = 224) -> transforms.Compose:
    # Resize the short side first so we do not crash on portrait phone
    # photos, then center-crop to a square that is a multiple of the
    # ViT-14 patch size.
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# Pre-baked transform variants exposed via the model name.
# - dinov2_vit*_hires uses 518x518 (37 * 14, DINOv2's native pre-train res).
# - The default 224x224 is faster but loses fine-grained detail (signage,
#   floor numbers) that matters for the IE Tower task.
_HIRES_SIZE = 518


def get_image_transform(model_name: str = "resnet50"):
    model_name = model_name.lower()

    if model_name == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V2.transforms()

    if model_name.endswith("_hires"):
        return _dinov2_transform(image_size=_HIRES_SIZE)

    if model_name.startswith("dinov2"):
        return _dinov2_transform()

    return build_default_transform()
