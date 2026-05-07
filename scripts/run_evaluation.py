import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import evaluate_retrieval
from src.utils.config import EMBEDDINGS_DIR, RESULTS_DIR
from src.utils.io import load_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VPR retrieval setup.")
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
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 5],
        help="List of K values used for Top-K accuracy.",
    )
    parser.add_argument(
        "--query-split",
        default="query",
        help="Value in the split column used for queries.",
    )
    parser.add_argument(
        "--gallery-split",
        default="gallery",
        help="Value in the split column used for gallery images.",
    )
    parser.add_argument(
        "--output-path",
        default=str(RESULTS_DIR / "evaluation.json"),
        help="Path for saving evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help=(
            "Run EasyOCR on every held-out query and add an `ocr` block "
            "to the JSON with ocr_coverage (queries that produced a label) "
            "and ocr_top_1_accuracy (correct of those). Off by default "
            "because the OCR pass is significantly slower than retrieval."
        ),
    )
    return parser.parse_args()


def _maybe_evaluate_ocr(query_metadata, project_root):
    """Run OCR on every held-out query path and aggregate coverage + accuracy."""
    from PIL import Image

    from src.features.ocr_predictor import OCRFloorPredictor

    if "image_path" not in query_metadata.columns:
        return None

    known = set(query_metadata["label"].astype(str).unique())
    predictor = OCRFloorPredictor(known_labels=known, confidence_threshold=0.6)
    if not predictor.is_available():
        return {
            "available": False,
            "error": "easyocr unavailable on this machine",
            "ocr_coverage": 0.0,
            "ocr_top_1_accuracy": 0.0,
        }

    considered = 0
    correct = 0
    coverage = 0
    raw_paths = query_metadata["image_path"].astype(str).tolist()
    truths = query_metadata["label"].astype(str).tolist()
    for raw_path, truth in zip(raw_paths, truths):
        path = Path(raw_path)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            alt = project_root / "data" / raw_path
            if alt.exists():
                path = alt
            else:
                continue
        try:
            image = Image.open(path)
        except Exception:
            continue
        considered += 1
        result = predictor.predict(image)
        if result.label is not None:
            coverage += 1
            if result.label == truth:
                correct += 1

    return {
        "available": True,
        "considered": considered,
        "ocr_coverage": coverage / considered if considered else 0.0,
        "ocr_top_1_accuracy": correct / coverage if coverage else 0.0,
        "ocr_overall_top_1": correct / considered if considered else 0.0,
    }


def main() -> None:
    args = parse_args()
    embeddings, metadata = load_embeddings(args.embeddings_path, args.metadata_path)

    use_split_evaluation = (
        "split" in metadata.columns
        and args.query_split in metadata["split"].astype(str).unique()
        and args.gallery_split in metadata["split"].astype(str).unique()
    )

    if use_split_evaluation:
        query_mask = metadata["split"].astype(str) == args.query_split
        gallery_mask = metadata["split"].astype(str) == args.gallery_split

        query_embeddings = embeddings[query_mask.to_numpy()]
        gallery_embeddings = embeddings[gallery_mask.to_numpy()]
        query_metadata = metadata.loc[query_mask].reset_index(drop=True)
        gallery_metadata = metadata.loc[gallery_mask].reset_index(drop=True)
    else:
        query_embeddings = None
        query_metadata = None
        gallery_embeddings = embeddings
        gallery_metadata = metadata

    metrics = evaluate_retrieval(
        gallery_embeddings=gallery_embeddings,
        gallery_metadata=gallery_metadata,
        query_embeddings=query_embeddings,
        query_metadata=query_metadata,
        top_ks=args.top_k,
        metric=args.metric,
    )

    if args.use_ocr and query_metadata is not None:
        ocr_metrics = _maybe_evaluate_ocr(query_metadata, PROJECT_ROOT)
        if ocr_metrics is not None:
            metrics["ocr"] = ocr_metrics

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()
