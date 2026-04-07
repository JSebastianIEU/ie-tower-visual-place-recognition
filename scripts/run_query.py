import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extract_embeddings import extract_single_image_embedding
from src.features.models import get_feature_extractor, resolve_device
from src.features.transforms import get_image_transform
from src.utils.config import EMBEDDINGS_DIR, INDEX_DIR
from src.utils.io import require_file
from src.retrieval.faiss_utils import load_index
from src.retrieval.search import search_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a FAISS VPR index.")
    parser.add_argument(
        "--image",
        "--query-image",
        dest="image",
        required=True,
        help="Path to the query image.",
    )
    parser.add_argument(
        "--index-path",
        default=str(INDEX_DIR / "gallery.index"),
        help="Path to the FAISS index.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(EMBEDDINGS_DIR / "gallery_metadata.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--model-name",
        default="resnet50",
        choices=["resnet50"],
        help="Backbone used to extract the query embedding.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, etc.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved images.")
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "l2"],
        help="Metric used by the index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    metadata = pd.read_csv(require_file(args.metadata_path, "Metadata CSV"))
    index = load_index(require_file(args.index_path, "FAISS index"))
    model, _ = get_feature_extractor(args.model_name)
    transform = get_image_transform(args.model_name)

    query_embedding = extract_single_image_embedding(
        image_path=require_file(args.image, "Query image"),
        model=model,
        transform=transform,
        device=device,
        normalize=args.metric == "cosine",
    )

    results = search_index(
        query_embeddings=query_embedding,
        index=index,
        metadata=metadata,
        top_k=args.top_k,
        metric=args.metric,
    )[0]

    print(f"Query image: {args.image}")
    for result in results:
        print(
            f"{result.rank:>2}. score={result.score:.4f} "
            f"label={result.label} path={result.image_path}"
        )


if __name__ == "__main__":
    main()
