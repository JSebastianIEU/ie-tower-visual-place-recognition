import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import ImagePlaceDataset
from src.features.extract_embeddings import extract_embeddings
from src.features.models import get_feature_extractor, resolve_device
from src.features.transforms import get_image_transform
from src.utils.config import DATA_DIR, DEFAULT_DATASET_CSV, EMBEDDINGS_DIR
from src.utils.io import save_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract image embeddings for VPR.")
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_DATASET_CSV),
        help="Path to dataset CSV.",
    )
    parser.add_argument(
        "--image-root",
        default=str(DATA_DIR),
        help="Optional root directory for relative image paths.",
    )
    parser.add_argument(
        "--model-name",
        default="resnet50",
        choices=["resnet50"],
        help="Pretrained backbone used for embedding extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="cpu, cuda, etc.")
    parser.add_argument(
        "--output-dir",
        default=str(EMBEDDINGS_DIR),
        help="Directory where embeddings and metadata will be stored.",
    )
    parser.add_argument(
        "--output-prefix",
        default="gallery",
        help="Prefix for the saved embeddings and metadata files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    transform = get_image_transform(args.model_name)
    dataset = ImagePlaceDataset(
        csv_path=args.csv_path,
        image_root=args.image_root,
        transform=transform,
    )
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
        output_dir=args.output_dir,
        prefix=args.output_prefix,
    )

    print(f"Saved {len(metadata)} embeddings with dimension {embedding_dim}.")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata:   {metadata_path}")


if __name__ == "__main__":
    main()
