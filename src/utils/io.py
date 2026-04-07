from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def require_file(path: str | Path, description: str = "File") -> Path:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return file_path


def save_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "gallery",
) -> Tuple[Path, Path]:
    output_dir = ensure_directory(output_dir)
    embeddings_path = output_dir / f"{prefix}_embeddings.npy"
    metadata_path = output_dir / f"{prefix}_metadata.csv"

    np.save(embeddings_path, embeddings.astype(np.float32))
    metadata.to_csv(metadata_path, index=False)
    return embeddings_path, metadata_path


def load_embeddings(
    embeddings_path: str | Path,
    metadata_path: str | Path,
) -> Tuple[np.ndarray, pd.DataFrame]:
    embeddings = np.load(require_file(embeddings_path, "Embeddings file")).astype(
        np.float32
    )
    metadata = pd.read_csv(require_file(metadata_path, "Metadata file"))
    return embeddings, metadata
