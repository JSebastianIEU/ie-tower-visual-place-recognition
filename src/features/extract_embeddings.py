from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _value_from_batch(batch_value, sample_index: int):
    value = batch_value[sample_index]
    if torch.is_tensor(value):
        return value.item()
    return value


def extract_embeddings(
    dataset: Dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    normalize: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run a dataset through a feature extractor and collect embeddings."""

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = model.to(device)
    model.eval()

    embedding_batches = []
    metadata_rows = []

    with torch.inference_mode():
        for batch in dataloader:
            images = batch["image"].to(device)
            batch_embeddings = model(images)

            if normalize:
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            embedding_batches.append(batch_embeddings.cpu().numpy().astype(np.float32))

            metadata_keys = [key for key in batch.keys() if key != "image"]
            batch_size_actual = len(batch["label"])
            for sample_index in range(batch_size_actual):
                metadata_rows.append(
                    {
                        key: _value_from_batch(batch[key], sample_index)
                        for key in metadata_keys
                    }
                )

    if not embedding_batches:
        raise ValueError("No embeddings were extracted because the dataset is empty.")

    embeddings = np.concatenate(embedding_batches, axis=0)
    metadata = pd.DataFrame(metadata_rows)
    return embeddings, metadata


def extract_single_image_embedding(
    image_path: str | Path,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    normalize: bool = True,
) -> np.ndarray:
    """Extract a single embedding from one query image."""

    image = Image.open(Path(image_path)).convert("RGB")
    return extract_pil_image_embedding(
        image=image,
        model=model,
        transform=transform,
        device=device,
        normalize=normalize,
    )


def extract_pil_image_embedding(
    image: Image.Image,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    normalize: bool = True,
) -> np.ndarray:
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        embedding = model(image_tensor)
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)
