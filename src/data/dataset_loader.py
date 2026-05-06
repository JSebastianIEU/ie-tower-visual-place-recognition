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
        split_filter: Optional[str] = None,
        skip_missing: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform
        self.split_filter = split_filter

        self.data = pd.read_csv(self.csv_path)
        missing_columns = REQUIRED_COLUMNS - set(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"Dataset CSV is missing required columns: {sorted(missing_columns)}"
            )

        # Optional split filtering: when the caller asks for a specific split
        # (e.g. "gallery") and the column exists with at least one matching row,
        # we keep only those rows. If the column is absent or has no matches we
        # silently keep every row so legacy CSVs without splits keep working.
        if split_filter is not None and "split" in self.data.columns:
            mask = self.data["split"].fillna("").astype(str) == split_filter
            if mask.any():
                self.data = self.data.loc[mask].reset_index(drop=True)

        # Optionally drop rows whose image_path does not resolve to an existing
        # file. Useful when running the pipeline on a partially-populated repo
        # (e.g. Ariel's floor10-16 rows but his frames were never committed).
        if skip_missing:
            existing_mask = self.data["image_path"].astype(str).apply(
                lambda p: self._resolve_image_path(p).exists()
            )
            dropped = (~existing_mask).sum()
            if dropped:
                print(
                    f"[dataset] skipping {dropped} row(s) with missing image files."
                )
            self.data = self.data.loc[existing_mask].reset_index(drop=True)

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
