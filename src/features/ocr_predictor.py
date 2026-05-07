"""Optional OCR head that short-circuits floor predictions when a number
is visible in the query photo.

Why we want it. The retrieval pipeline confuses `floor10_hallway_left`
with `floor15_hallway_left` because the IE Tower repeats the same layout
on every above-ground floor. But if the query image contains a visible
floor-number plaque (the actual number painted on the wall, the panel
above the elevator door, etc.), reading that number directly is much
more reliable than the visual retrieval. This module wraps EasyOCR and
exposes a single ``OCRFloorPredictor.predict(image)`` API.

Why a separate module. EasyOCR is heavy (~150 MB of weights downloaded
on first use) and we do not want it loaded just to import the package.
``OCRFloorPredictor`` lazy-loads the reader on first ``predict`` call
and gracefully reports "OCR unavailable" if the import fails.

The predictor returns ``label = "floorN"`` or ``"basementN"`` only when:
  1. EasyOCR is available.
  2. At least one detected text region matches the floor-number regex.
  3. The detection confidence is above ``confidence_threshold`` (default 0.6).
  4. The candidate label appears in ``known_labels`` (so we never invent
     a floor that the gallery does not cover).

The third return value (``raw_detections``) is always populated even
when no usable label is found — useful for debugging and for the demo
to render the OCR text it saw.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageOps


# Match either ``B<N>`` (basement) or a bare 1-2 digit number that looks
# like a floor plaque. We deliberately do NOT match longer digit strings
# (room numbers like "4.01" or "203b") because those are not floor
# numbers and would generate false positives.
_FLOOR_NUMBER_REGEX = re.compile(r"^(?:B(?P<basement>\d)|0?(?P<floor>\d{1,2}))$", re.IGNORECASE)


@dataclass
class OCRDetection:
    """One text region returned by EasyOCR."""

    text: str
    confidence: float
    raw_text: str

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
        }


@dataclass
class OCRPrediction:
    """Result of running ``OCRFloorPredictor.predict`` on one image."""

    label: Optional[str]
    confidence: float
    detections: list[OCRDetection] = field(default_factory=list)
    available: bool = True
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "detections": [d.as_dict() for d in self.detections],
            "available": self.available,
            "error": self.error,
        }


class OCRFloorPredictor:
    """EasyOCR wrapper that maps detected numbers to gallery floor labels.

    Parameters
    ----------
    known_labels:
        Iterable of label strings that the gallery covers
        (``floor3``, ``floor10``, ``basement0``, …). Detections whose
        normalised label is not in this set are dropped — that way OCR
        cannot invent a floor that the index can't return.
    confidence_threshold:
        Minimum EasyOCR per-detection confidence required to accept a
        candidate. Defaults to 0.6 to mirror the spatial-chain default.
    languages:
        Forwarded to ``easyocr.Reader``. English-only is enough for the
        IE Tower signage we see today.
    """

    def __init__(
        self,
        known_labels,
        confidence_threshold: float = 0.6,
        languages: tuple[str, ...] = ("en",),
    ) -> None:
        self.known_labels = {str(lbl).strip() for lbl in known_labels if lbl}
        self.confidence_threshold = confidence_threshold
        self._languages = list(languages)
        self._reader = None
        self._available: Optional[bool] = None
        self._import_error: Optional[str] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Probe whether ``easyocr`` is importable. Cached."""
        if self._available is not None:
            return self._available
        with self._lock:
            if self._available is not None:
                return self._available
            try:
                import easyocr  # noqa: F401  (probe only)
                self._available = True
            except Exception as exc:  # noqa: BLE001
                self._available = False
                self._import_error = f"easyocr import failed: {exc}"
        return self._available

    def predict(self, image: Image.Image) -> OCRPrediction:
        """Run OCR on a PIL image and try to extract a floor label.

        On any error the predictor returns ``available=False`` /
        ``label=None`` so the caller can degrade gracefully (skip the OCR
        path and use retrieval only) without try/except glue.
        """
        if not self.is_available():
            return OCRPrediction(
                label=None,
                confidence=0.0,
                detections=[],
                available=False,
                error=self._import_error or "easyocr unavailable",
            )

        try:
            reader = self._ensure_reader()
        except Exception as exc:  # noqa: BLE001
            return OCRPrediction(
                label=None,
                confidence=0.0,
                detections=[],
                available=False,
                error=f"easyocr Reader init failed: {exc}",
            )

        # Apply EXIF rotation up-front — the demo / notebook already do
        # this on uploads, but predict() may also be called from offline
        # paths.
        image = ImageOps.exif_transpose(image).convert("RGB")
        np_image = np.asarray(image)

        try:
            raw = reader.readtext(np_image)
        except Exception as exc:  # noqa: BLE001
            return OCRPrediction(
                label=None,
                confidence=0.0,
                detections=[],
                available=False,
                error=f"easyocr readtext failed: {exc}",
            )

        detections: list[OCRDetection] = []
        candidates: list[tuple[str, float]] = []  # (label, confidence)
        for entry in raw:
            # easyocr.readtext returns [bbox, text, confidence].
            try:
                _, text, confidence = entry
            except Exception:  # noqa: BLE001
                continue
            text = (text or "").strip()
            confidence = float(confidence)
            mapped = self._map_to_label(text)
            detection = OCRDetection(
                text=mapped or "",
                confidence=confidence,
                raw_text=text,
            )
            detections.append(detection)
            if (
                mapped
                and mapped in self.known_labels
                and confidence >= self.confidence_threshold
            ):
                candidates.append((mapped, confidence))

        if not candidates:
            return OCRPrediction(
                label=None,
                confidence=0.0,
                detections=detections,
                available=True,
            )

        # If multiple text regions agree on a label, take the highest-
        # confidence one. If they disagree, pick the most-confident
        # detection — but only commit to a label if it beats the
        # threshold by a clear margin so we don't oscillate between
        # adjacent floor numbers when the photo contains both (rare on
        # this dataset, but possible on signage panels).
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        top_label, top_confidence = candidates[0]
        return OCRPrediction(
            label=top_label,
            confidence=top_confidence,
            detections=detections,
            available=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_reader(self):
        if self._reader is not None:
            return self._reader
        with self._lock:
            if self._reader is not None:
                return self._reader
            import easyocr  # type: ignore[import-not-found]

            self._reader = easyocr.Reader(self._languages, gpu=False, verbose=False)
        return self._reader

    @staticmethod
    def _map_to_label(text: str) -> Optional[str]:
        """Map a free-form OCR string to a floor label or ``None``.

        Strips common adornments (parentheses, dots, "F" prefixes) and
        runs the floor-number regex over what remains.
        """
        if not text:
            return None
        cleaned = text.strip().upper()
        # Drop trailing/leading punctuation that tends to surround signage.
        cleaned = cleaned.strip(" .,()[]/\\|:;\"'")
        # Strip leading "F" or "FL" or "FLOOR" — common on plaques.
        for prefix in ("FLOOR", "FLR", "FL", "F"):
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
                rest = cleaned[len(prefix):].lstrip("0 .-")
                if rest.isdigit() or (rest.startswith("0") and rest[1:].isdigit()):
                    cleaned = rest
                    break
        match = _FLOOR_NUMBER_REGEX.match(cleaned)
        if not match:
            return None
        if match.group("basement") is not None:
            return f"basement{int(match.group('basement'))}"
        return f"floor{int(match.group('floor'))}"
