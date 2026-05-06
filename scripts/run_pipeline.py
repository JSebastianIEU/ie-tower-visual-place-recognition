"""Run the basic VPR pipeline end-to-end.

What this script does:
1. Loads every row of the dataset CSV (no split filtering at this stage).
2. Extracts ResNet50 embeddings for every row.
3. Saves embeddings + metadata under ``outputs/embeddings/``.
4. Builds a FAISS index from the same embeddings and saves it under
   ``outputs/index/``.

Why we don't filter to ``split == 'gallery'`` here:

* ``run_evaluation.py`` rebuilds its own FAISS index from gallery rows only
  (see ``src/evaluation/evaluate.py``). Filtering at this stage would not
  improve evaluation correctness.
* The Marimo demo searches the saved index. Having every available frame
  available as a candidate is *better* UX, not worse — the user uploads a
  photo, gets the closest frames, and each frame still carries the right
  ``label``. Excluding query frames would only shrink the reference set.

Reproducibility: ``set_global_seed(42)`` is called at startup so re-running on
the same data produces identical embeddings and an identical index.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import ImagePlaceDataset
from src.features.extract_embeddings import extract_embeddings
from src.features.models import get_feature_extractor, resolve_device
from src.features.transforms import get_image_transform
from src.retrieval.build_index import build_faiss_index
from src.retrieval.faiss_utils import save_index
from src.utils.config import DATA_DIR, DEFAULT_DATASET_CSV, EMBEDDINGS_DIR, INDEX_DIR
from src.utils.io import save_embeddings


SEED = 42


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so feature extraction is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the basic VPR pipeline end to end.")
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_DATASET_CSV),
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--image-root",
        default=str(DATA_DIR),
        help="Root directory for images.",
    )
    parser.add_argument(
        "--model-name",
        default="resnet50",
        choices=[
            "resnet50",
            "dinov2_vits14",
            "dinov2_vits14_hires",
            "dinov2_vitb14",
            "dinov2_vitb14_hires",
            "dinov2_vitl14",
        ],
        help=(
            "Backbone used for feature extraction. resnet50 is the ImageNet "
            "baseline; dinov2_vit* uses Meta AI's self-supervised ViT family "
            "which is empirically stronger on retrieval (S=384d, B=768d, "
            "L=1024d)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, etc.")
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "l2"],
        help="Metric used to build the FAISS index.",
    )
    parser.add_argument(
        "--embedding-output-dir",
        default=str(EMBEDDINGS_DIR),
    )
    parser.add_argument(
        "--output-prefix",
        default="gallery",
        help=(
            "Prefix for the saved embedding/metadata files. Kept as 'gallery' "
            "for backward compatibility even though the saved files include "
            "every row of the CSV."
        ),
    )
    parser.add_argument(
        "--index-path",
        default=str(INDEX_DIR / "gallery.index"),
        help="Where the FAISS index will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(SEED)

    device = resolve_device(args.device)
    transform = get_image_transform(args.model_name)
    dataset = ImagePlaceDataset(
        csv_path=args.csv_path,
        image_root=args.image_root,
        transform=transform,
        skip_missing=True,
    )
    print(f"[pipeline] Loaded {len(dataset)} rows from {args.csv_path}")

    model, embedding_dim = get_feature_extractor(args.model_name)

    embeddings, metadata = extract_embeddings(
        dataset=dataset,
        model=model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    embeddings_path, metadata_path = save_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        output_dir=args.embedding_output_dir,
        prefix=args.output_prefix,
    )

    # Side-car file so downstream consumers (the Marimo demo, the Jupyter
    # notebook, anything offline) know which backbone produced the
    # embeddings without having to inspect the .npy shape and guess.
    info = {
        "model_name": args.model_name,
        "embedding_dim": int(embedding_dim),
        "metric": args.metric,
        "num_rows": int(len(metadata)),
        "seed": SEED,
    }
    info_path = Path(args.embedding_output_dir) / f"{args.output_prefix}_info.json"
    info_path.write_text(__import__("json").dumps(info, indent=2), encoding="utf-8")

    index = build_faiss_index(embeddings=embeddings, metric=args.metric)
    index_path = save_index(index, args.index_path)

    print(f"Processed {len(metadata)} images into {embedding_dim}-D embeddings.")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"Index:      {index_path}")


if __name__ == "__main__":
    main()
