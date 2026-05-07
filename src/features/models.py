"""Feature-extractor backbones for the VPR pipeline.

Two families are exposed:

* ``resnet50`` — the original baseline. Frozen, ImageNet-pretrained,
  outputs 2048-d global features (the average pool before the final FC
  layer).
* ``dinov2_vits14`` / ``dinov2_vitb14`` / ``dinov2_vitl14`` — Meta AI's
  self-supervised ViT family. Pretrained on LVD-142M, frozen, returns the
  ``[CLS]`` token (384 / 768 / 1024 dim respectively). Empirically these
  give a large boost on retrieval tasks because the model was trained
  with discrimination as the objective rather than ImageNet
  classification.

All backbones return a single feature vector per image so the rest of
the pipeline (FAISS index, search, evaluation, demo) is unchanged.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


_DINOV2_HUB_NAMES = {
    "dinov2_vits14": ("dinov2_vits14", 384),
    "dinov2_vits14_hires": ("dinov2_vits14", 384),
    "dinov2_vitb14": ("dinov2_vitb14", 768),
    "dinov2_vitb14_hires": ("dinov2_vitb14", 768),
    "dinov2_vitl14": ("dinov2_vitl14", 1024),
}


class GlobalFeatureExtractor(nn.Module):
    """Wrap a CNN backbone and expose a flattened embedding vector."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return torch.flatten(features, start_dim=1)


class DinoV2FeatureExtractor(nn.Module):
    """Adapt a torch.hub-loaded DINOv2 backbone to our pipeline.

    DINOv2's forward returns the [CLS] token features directly, but
    ``forward_features`` exposes the full set of patch tokens. We use the
    [CLS] token because it is the canonical global descriptor used in the
    DINOv2 paper for image-level retrieval / classification probes.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Newer DINOv2 checkpoints return the CLS token from forward(); fall
        # back to ``forward_features`` if forward returns a dict.
        out = self.backbone(images)
        if isinstance(out, dict):
            for key in ("x_norm_clstoken", "x_clstoken", "cls_token"):
                if key in out:
                    return out[key]
            raise RuntimeError(f"Unexpected DINOv2 output keys: {list(out.keys())}")
        if out.dim() == 3:
            # (batch, tokens, dim) — first token is CLS by convention.
            return out[:, 0]
        return out


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_resnet50(pretrained: bool) -> tuple[nn.Module, int]:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    backbone = resnet50(weights=weights)
    embedding_dim = backbone.fc.in_features
    feature_layers = list(backbone.children())[:-1]
    model = GlobalFeatureExtractor(nn.Sequential(*feature_layers))
    model.eval()
    return model, embedding_dim


def _load_dinov2(model_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    hub_name, embedding_dim = _DINOV2_HUB_NAMES[model_name]
    if not pretrained:
        raise ValueError("DINOv2 backbones only ship with pretrained weights.")
    # ``torch.hub.load`` caches the checkpoint under ~/.cache/torch/hub/.
    # ``trust_repo=True`` skips the y/n prompt on first run.
    backbone = torch.hub.load(
        "facebookresearch/dinov2",
        hub_name,
        trust_repo=True,
        verbose=False,
    )
    backbone.eval()
    model = DinoV2FeatureExtractor(backbone)
    model.eval()
    return model, embedding_dim


def get_feature_extractor(
    model_name: str = "resnet50",
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    model_name = model_name.lower()

    if model_name == "resnet50":
        return _load_resnet50(pretrained)

    if model_name in _DINOV2_HUB_NAMES:
        return _load_dinov2(model_name, pretrained)

    available = ["resnet50", *sorted(_DINOV2_HUB_NAMES)]
    raise ValueError(
        f"Unsupported model '{model_name}'. Available options: {available}"
    )


class ProjectionHead(nn.Module):
    """Small 2-layer MLP that re-projects frozen features into a tighter
    embedding space tuned for floor-level retrieval.

    Architecture: ``in_dim → hidden → out_dim`` with BatchNorm + ReLU
    between the layers. The output is L2-normalised so cosine search
    against a FAISS Flat-IP index keeps working.

    The default hidden = 256 and out_dim = 128 are sized for the
    DINOv2 ViT-S 384-d input at our dataset scale (~2k gallery rows): big
    enough to fit a meaningful manifold, small enough to train on CPU in
    a few minutes.
    """

    def __init__(
        self,
        in_dim: int = 384,
        hidden_dim: int = 256,
        out_dim: int = 128,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # When ``residual`` is True the head must keep the same dim as the
        # input so we can add the input back. This biases initialization
        # toward "identity + small perturbation" — useful when the frozen
        # features are already strong and we just want to nudge them.
        self.residual = residual
        if residual:
            assert out_dim == in_dim, (
                f"residual heads require out_dim == in_dim, got "
                f"out_dim={out_dim}, in_dim={in_dim}"
            )
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        if residual:
            # Initialise the last layer near zero so the head starts as
            # an identity-ish mapping (output ≈ x).
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.layers(x)
        if self.residual:
            out = x + delta
        else:
            out = delta
        return torch.nn.functional.normalize(out, dim=1)


class DinoV2WithProjectionHead(nn.Module):
    """Wraps a frozen DINOv2 backbone and a trainable ``ProjectionHead``.

    The backbone is set to ``eval`` and its parameters are frozen at
    construction time. ``forward`` projects the image through the
    backbone, applies the head, and returns the L2-normalised output.

    Usage::

        backbone, _ = get_feature_extractor("dinov2_vits14_hires")
        head = ProjectionHead(in_dim=384)
        head.load_state_dict(torch.load(checkpoint))
        model = DinoV2WithProjectionHead(backbone, head)
        model.eval()

    The head is intentionally separate from the backbone so the training
    script can swap heads (different hidden dims, different margins)
    without re-loading DINOv2 every time.
    """

    def __init__(self, backbone: nn.Module, head: ProjectionHead) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(images)
        return self.head(features)
