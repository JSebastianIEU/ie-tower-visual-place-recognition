import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.build_index import build_faiss_index
from src.retrieval.faiss_utils import save_index
from src.utils.config import EMBEDDINGS_DIR, INDEX_DIR
from src.utils.io import load_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from embeddings.")
    parser.add_argument(
        "--embeddings-path",
        default=str(EMBEDDINGS_DIR / "gallery_embeddings.npy"),
        help="Path to .npy embeddings.",
    )
    parser.add_argument(
        "--metadata-path",
        default=str(EMBEDDINGS_DIR / "gallery_metadata.csv"),
        help="Path to metadata CSV.",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "l2"],
        help="Similarity metric used by FAISS.",
    )
    parser.add_argument(
        "--index-path",
        default=str(INDEX_DIR / "gallery.index"),
        help="Output path for the FAISS index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings, metadata = load_embeddings(args.embeddings_path, args.metadata_path)
    index = build_faiss_index(embeddings=embeddings, metric=args.metric)
    saved_path = save_index(index, args.index_path)

    print(f"Indexed {len(metadata)} vectors.")
    print(f"Index saved to: {saved_path}")


if __name__ == "__main__":
    main()
