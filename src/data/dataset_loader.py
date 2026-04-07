from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"image_path", "label"}


class ImagePlaceDataset(Dataset):
    """Simple dataset for place-recognition images described in a CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path | None = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform

        self.data = pd.read_csv(self.csv_path)
        missing_columns = REQUIRED_COLUMNS - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset CSV is missing required columns: {sorted(missing_columns)}"
            )

        self.extra_columns = [
            column for column in self.data.columns if column not in REQUIRED_COLUMNS
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        image_path = self._resolve_image_path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        sample = {
            "image": image,
            "label": str(row["label"]),
            "image_path": str(image_path),
            "index": index,
        }

        for column in self.extra_columns:
            sample[column] = row[column]

        return sample

    def get_metadata(self) -> pd.DataFrame:
        metadata = self.data.copy()
        metadata["image_path"] = metadata["image_path"].apply(
            lambda value: str(self._resolve_image_path(str(value)))
        )
        return metadata

    def _resolve_image_path(self, image_path: str) -> Path:
        candidate = Path(image_path)
        if candidate.is_absolute():
            return candidate

        search_roots = []
        if self.image_root is not None:
            search_roots.append(self.image_root)
        search_roots.extend(
            [
                self.csv_path.parent,
                self.csv_path.parent.parent,
                self.csv_path.parent.parent.parent,
            ]
        )

        for root in search_roots:
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved

        if self.image_root is not None:
            return (self.image_root / candidate).resolve()

        return (self.csv_path.parent / candidate).resolve()
