"""IE Tower VPR — interactive Marimo dashboard.

Three sections:

1. **Dataset overview & metrics** — total frames, per-floor counts,
   evaluation results from ``outputs/results/evaluation.json`` (if present),
   and a 2-D PCA projection of the gallery embeddings coloured by floor.
2. **Test the model** — upload your own photo OR press "Pick a random query"
   to test against a held-out query frame.
3. **Top-K results** — thumbnails of the closest gallery frames with scores
   and labels.

Open with: ``marimo edit app/demo.py``
"""
import marimo

__generated_with = "0.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import io
    import json
    import mimetypes
    import random
    import sys
    from collections import Counter
    from html import escape
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.features.extract_embeddings import extract_pil_image_embedding
    from src.features.models import get_feature_extractor, resolve_device
    from src.features.transforms import get_image_transform
    from src.retrieval.faiss_utils import load_index
    from src.retrieval.search import search_index
    from src.utils.config import EMBEDDINGS_DIR, INDEX_DIR, RESULTS_DIR, ROOT

    return (
        Counter,
        EMBEDDINGS_DIR,
        INDEX_DIR,
        Image,
        ImageOps,
        Path,
        PROJECT_ROOT,
        RESULTS_DIR,
        ROOT,
        base64,
        escape,
        extract_pil_image_embedding,
        get_feature_extractor,
        get_image_transform,
        io,
        json,
        load_index,
        mimetypes,
        mo,
        np,
        pd,
        plt,
        random,
        resolve_device,
        search_index,
    )


@app.cell
def _(Counter, Image, ImageOps, Path, base64, escape, io, mimetypes):
    # --- Image upload + decoding helpers -----------------------------------

    def read_uploaded_image(upload_value):
        if not upload_value:
            return None, None, "Upload a JPG or PNG image to start a search."

        item = upload_value[0] if isinstance(upload_value, (list, tuple)) else upload_value
        filename = getattr(item, "name", None)
        payload = None

        if isinstance(item, dict):
            filename = item.get("name", filename)
            for key in ("contents", "content", "data", "bytes"):
                if item.get(key) is not None:
                    payload = item[key]
                    break
        elif isinstance(item, (bytes, bytearray)):
            payload = bytes(item)
        else:
            for attr in ("contents", "content", "data", "bytes"):
                if hasattr(item, attr):
                    payload = getattr(item, attr)
                    break
            if payload is None and hasattr(item, "read"):
                payload = item.read()

        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if isinstance(payload, bytearray):
            payload = bytes(payload)
        if isinstance(payload, str):
            if ";base64," in payload:
                payload = base64.b64decode(payload.split(";base64,", 1)[1])
            else:
                payload = payload.encode("utf-8")
        if isinstance(payload, list) and payload and isinstance(payload[0], int):
            payload = bytes(payload)

        if not isinstance(payload, bytes):
            return None, None, "The uploaded file could not be read as image data."

        try:
            image = Image.open(io.BytesIO(payload))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            return None, None, "The uploaded file is not a valid image."

        return image, payload, filename or "query.jpg"

    def bytes_to_data_uri(data, filename):
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "image/jpeg"
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def path_to_data_uri(image_path):
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return None
        return bytes_to_data_uri(path.read_bytes(), path.name)

    # --- Result rendering ---------------------------------------------------

    def render_result_card(rank, score, label, image_path, data_uri, ground_truth=None):
        label_text = escape(label) if label else "N/A"
        path_text = escape(image_path) if image_path else "N/A"
        # Highlight matches with green outline, mismatches with red.
        border = "#d1d5db"
        badge = ""
        if ground_truth is not None and label:
            if label == ground_truth:
                border = "#10b981"
                badge = (
                    '<span style="background:#10b981;color:white;padding:2px 8px;'
                    'border-radius:8px;font-size:11px;margin-left:8px;">match</span>'
                )
            else:
                border = "#ef4444"
                badge = (
                    '<span style="background:#ef4444;color:white;padding:2px 8px;'
                    'border-radius:8px;font-size:11px;margin-left:8px;">miss</span>'
                )

        image_html = (
            f'<img src="{data_uri}" alt="Result {rank}" '
            'style="width:100%;height:180px;object-fit:cover;border-radius:8px;" />'
            if data_uri
            else (
                '<div style="height:180px;display:flex;align-items:center;'
                'justify-content:center;background:#f3f4f6;border-radius:8px;'
                'color:#6b7280;">Preview unavailable</div>'
            )
        )

        return f"""
        <div style="border:2px solid {border};border-radius:12px;padding:12px;background:#ffffff;">
          <div style="font-size:14px;font-weight:600;margin-bottom:8px;">Rank {rank}{badge}</div>
          {image_html}
          <div style="margin-top:10px;font-size:14px;"><strong>Score:</strong> {score:.4f}</div>
          <div style="font-size:14px;"><strong>Label:</strong> {label_text}</div>
          <div style="font-size:11px;color:#6b7280;word-break:break-word;margin-top:6px;">{path_text}</div>
        </div>
        """

    def summarize_prediction(results):
        if not results:
            return None, 0, 0
        labels = [r.label for r in results if r.label]
        if not labels:
            return None, 0, len(results)
        counts = Counter(labels)
        top_count = max(counts.values())
        for r in results:
            if counts.get(r.label, 0) == top_count:
                return r.label, top_count, len(results)
        return None, 0, len(results)

    def render_prediction_banner(predicted_label, votes, total, ground_truth=None):
        if not predicted_label or total == 0:
            return (
                '<div style="padding:14px;border-radius:12px;background:#fef3c7;'
                'color:#92400e;font-size:14px;">No prediction available.</div>'
            )
        confidence = votes / total
        # If ground truth is known, colour-code the banner.
        if ground_truth is not None:
            if predicted_label == ground_truth:
                bg, fg, border = "#ecfdf5", "#065f46", "#6ee7b7"
                tag = "✓ correct"
            else:
                bg, fg, border = "#fef2f2", "#991b1b", "#fca5a5"
                tag = f"✗ wrong (true: {escape(ground_truth)})"
        else:
            bg, fg, border = "#eff6ff", "#1e40af", "#93c5fd"
            tag = "no ground truth"

        return (
            f'<div style="padding:18px;border-radius:12px;background:{bg};'
            f'border:1px solid {border};color:{fg};font-size:18px;font-weight:600;">'
            f"Predicted floor: {escape(predicted_label)} "
            f'<span style="font-weight:400;font-size:14px;">'
            f"(confidence {votes}/{total} ≈ {confidence:.0%}) — {tag}</span></div>"
        )

    def render_query_preview(mo_module, image, data_uri, ground_truth=None):
        gt_html = ""
        if ground_truth is not None:
            gt_html = (
                f'<div style="margin-top:8px;font-size:14px;color:#374151;">'
                f"Ground truth: <strong>{escape(ground_truth)}</strong></div>"
            )
        return mo_module.Html(
            f'<div><img src="{data_uri}" alt="Query" '
            'style="max-width:320px;width:100%;border-radius:12px;border:1px solid #d1d5db;" />'
            f"{gt_html}</div>"
        )

    return (
        bytes_to_data_uri,
        path_to_data_uri,
        read_uploaded_image,
        render_prediction_banner,
        render_query_preview,
        render_result_card,
        summarize_prediction,
    )


@app.cell
def _(EMBEDDINGS_DIR, INDEX_DIR, RESULTS_DIR, json, load_index, np, pd):
    # --- Load gallery + eval artifacts --------------------------------------
    expected_index = INDEX_DIR / "gallery.index"
    expected_metadata = EMBEDDINGS_DIR / "gallery_metadata.csv"
    expected_embeddings = EMBEDDINGS_DIR / "gallery_embeddings.npy"
    expected_info = EMBEDDINGS_DIR / "gallery_info.json"
    expected_eval = RESULTS_DIR / "evaluation.json"
    expected_compare = ROOT / "outputs" / "_compare" / "comparison.json"

    errors = []
    index = None
    metadata = None
    embeddings = None
    eval_results = None
    info = None
    compare_results = None

    if not expected_index.exists():
        errors.append(f"Missing FAISS index: {expected_index}")
    else:
        index = load_index(expected_index)

    if not expected_metadata.exists():
        errors.append(f"Missing metadata CSV: {expected_metadata}")
    else:
        metadata = pd.read_csv(expected_metadata)

    if expected_embeddings.exists():
        embeddings = np.load(expected_embeddings).astype(np.float32)

    if expected_eval.exists():
        try:
            eval_results = json.loads(expected_eval.read_text(encoding="utf-8"))
        except Exception:
            eval_results = None

    if expected_info.exists():
        try:
            info = json.loads(expected_info.read_text(encoding="utf-8"))
        except Exception:
            info = None

    if expected_compare.exists():
        try:
            compare_results = json.loads(expected_compare.read_text(encoding="utf-8"))
        except Exception:
            compare_results = None

    return (
        compare_results,
        embeddings,
        errors,
        eval_results,
        expected_eval,
        expected_index,
        expected_metadata,
        index,
        info,
        metadata,
    )


@app.cell
def _(embeddings, get_feature_extractor, get_image_transform, info, resolve_device):
    # Pick the feature extractor that produced the gallery embeddings.
    # Priority: gallery_info.json (written by run_pipeline.py) > inferred
    # from embedding dim > resnet50 fallback.
    device = resolve_device(None)
    if info and info.get("model_name"):
        model_name = info["model_name"]
    elif embeddings is not None:
        dim_to_name = {
            2048: "resnet50",
            384: "dinov2_vits14",
            768: "dinov2_vitb14",
            1024: "dinov2_vitl14",
        }
        model_name = dim_to_name.get(int(embeddings.shape[1]), "resnet50")
    else:
        model_name = "resnet50"
    model, _ = get_feature_extractor(model_name)
    transform = get_image_transform(model_name)
    return device, model, model_name, transform


@app.cell(hide_code=True)
def _(errors, mo):
    if errors:
        body = "\n".join(f"- {m}" for m in errors)
        setup_view = mo.md(
            "# IE Tower VPR — Setup required\n\n"
            f"{body}\n\n"
            "Run `python scripts/run_all.py` first to build the artifacts, "
            "then reload this notebook."
        )
    else:
        setup_view = None
    return (setup_view,)


@app.cell(hide_code=True)
def _(setup_view):
    setup_view  # rendered only when artifacts are missing
    return


@app.cell(hide_code=True)
def _(mo, metadata):
    # --- Section 1: Dataset overview ---------------------------------------
    if metadata is None:
        overview_view = None
    else:
        n_rows = len(metadata)
        n_floors = metadata["label"].nunique()
        n_basements = sum(1 for x in metadata["label"].unique() if str(x).startswith("basement"))
        n_above = n_floors - n_basements
        if "split" in metadata.columns:
            split_counts = metadata["split"].fillna("").value_counts().to_dict()
            n_gallery = split_counts.get("gallery", 0)
            n_query = split_counts.get("query", 0)
        else:
            n_gallery = n_rows
            n_query = 0

        _cards_html = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:12px 0;">'
            f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;padding:14px;">'
            f'<div style="font-size:24px;font-weight:700;color:#1e3a8a;">{n_rows:,}</div>'
            f'<div style="font-size:13px;color:#374151;">total frames</div></div>'
            f'<div style="background:#ecfdf5;border:1px solid #6ee7b7;border-radius:12px;padding:14px;">'
            f'<div style="font-size:24px;font-weight:700;color:#065f46;">{n_floors}</div>'
            f'<div style="font-size:13px;color:#374151;">distinct floors '
            f'<span style="color:#6b7280;">({n_above} above, {n_basements} basement)</span></div></div>'
            f'<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:12px;padding:14px;">'
            f'<div style="font-size:24px;font-weight:700;color:#92400e;">{n_gallery:,}</div>'
            f'<div style="font-size:13px;color:#374151;">gallery rows (indexed)</div></div>'
            f'<div style="background:#f3e8ff;border:1px solid #d8b4fe;border-radius:12px;padding:14px;">'
            f'<div style="font-size:24px;font-weight:700;color:#6b21a8;">{n_query:,}</div>'
            f'<div style="font-size:13px;color:#374151;">held-out queries</div></div>'
            "</div>"
        )

        overview_view = mo.vstack(
            [
                mo.md("# IE Tower VPR — Interactive demo"),
                mo.md(
                    "ResNet50 (frozen, pretrained) extracts 2048-d features for every "
                    "frame in the gallery, FAISS does cosine retrieval, and the "
                    "predicted floor is the majority label across the Top-K matches."
                ),
                mo.md("## 1. Dataset overview"),
                mo.Html(_cards_html),
            ]
        )
    return (overview_view,)


@app.cell(hide_code=True)
def _(overview_view):
    overview_view
    return


@app.cell(hide_code=True)
def _(escape, info, mo, model_name):
    # --- Model card ---------------------------------------------------------
    # Show which backbone built the index that the demo is searching against.
    description = {
        "resnet50": "ResNet50 — ImageNet-pretrained CNN, 2048-d global features. Frozen; the original baseline.",
        "dinov2_vits14": "DINOv2 ViT-S/14 — Meta AI's self-supervised ViT, 384-d. 224x224 input, fast and strong on visual retrieval.",
        "dinov2_vits14_hires": "DINOv2 ViT-S/14 at 518x518 — DINOv2's native pre-train resolution. Slower but captures fine-grained signage detail.",
        "dinov2_vitb14": "DINOv2 ViT-B/14 — bigger DINOv2 (768-d). About 3x slower than S/14 on CPU and only marginally better on this dataset.",
        "dinov2_vitb14_hires": "DINOv2 ViT-B/14 at 518x518 — biggest model + native res. Even slower; best ceiling.",
        "dinov2_vitl14": "DINOv2 ViT-L/14 — 1024-d ViT-Large. Significant compute cost; usually overkill on CPU.",
    }.get(model_name, "Custom backbone.")
    extra = ""
    if info:
        extra = (
            f' &nbsp;·&nbsp; embedding_dim = {info.get("embedding_dim", "?")} '
            f'&nbsp;·&nbsp; metric = {escape(str(info.get("metric", "cosine")))} '
            f'&nbsp;·&nbsp; gallery rows = {info.get("num_rows", "?")}'
        )
    model_card_view = mo.Html(
        f'<div style="margin:8px 0 16px 0;padding:14px;border-radius:12px;'
        f'background:#f8fafc;border:1px solid #e2e8f0;font-size:14px;color:#1f2937;">'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:4px;">Model in use</div>'
        f'<div style="font-weight:700;font-size:16px;margin-bottom:6px;">'
        f'{escape(model_name)}</div>'
        f'<div style="color:#374151;">{escape(description)}</div>'
        f'<div style="margin-top:8px;font-size:12px;color:#6b7280;">{extra}</div>'
        f'</div>'
    )
    return (model_card_view,)


@app.cell(hide_code=True)
def _(model_card_view):
    model_card_view
    return


@app.cell(hide_code=True)
def _(eval_results, mo):
    # --- Eval metrics card --------------------------------------------------
    if not eval_results:
        eval_view = mo.md(
            "_No evaluation results found yet — run `python scripts/run_evaluation.py` to populate this section._"
        )
    else:
        # Pull common keys defensively.
        top1 = eval_results.get("top_1_accuracy")
        top5 = eval_results.get("top_5_accuracy")
        mean_ap = eval_results.get("mAP")

        def _pct(x):
            return f"{x:.1%}" if isinstance(x, (int, float)) else "—"

        _eval_html = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:8px 0 16px 0;">'
            f'<div style="background:#fff7ed;border:1px solid #fdba74;border-radius:12px;padding:14px;">'
            f'<div style="font-size:28px;font-weight:700;color:#9a3412;">{_pct(top1)}</div>'
            f'<div style="font-size:13px;color:#374151;">Top-1 accuracy</div></div>'
            f'<div style="background:#fff7ed;border:1px solid #fdba74;border-radius:12px;padding:14px;">'
            f'<div style="font-size:28px;font-weight:700;color:#9a3412;">{_pct(top5)}</div>'
            f'<div style="font-size:13px;color:#374151;">Top-5 accuracy</div></div>'
            f'<div style="background:#fff7ed;border:1px solid #fdba74;border-radius:12px;padding:14px;">'
            f'<div style="font-size:28px;font-weight:700;color:#9a3412;">{_pct(mean_ap)}</div>'
            f'<div style="font-size:13px;color:#374151;">mAP</div></div>'
            "</div>"
        )
        eval_view = mo.vstack([mo.md("### Evaluation metrics"), mo.Html(_eval_html)])
    return (eval_view,)


@app.cell(hide_code=True)
def _(eval_view):
    eval_view
    return


@app.cell(hide_code=True)
def _(compare_results, escape, mo, model_name):
    # --- Backbone comparison table ----------------------------------------
    # Reads outputs/_compare/comparison.json (written by the bake-off run).
    # Highlights the row corresponding to the backbone in use right now.
    if not compare_results:
        compare_view = None
    else:
        rows = []
        for backbone, payload in compare_results.items():
            if "top_1_accuracy" not in payload:
                continue
            highlight = backbone == model_name
            row_style = (
                ' style="background:#dcfce7;font-weight:600;"' if highlight else ""
            )
            rows.append(
                f"<tr{row_style}>"
                f'<td style="padding:8px 14px;">{escape(backbone)}'
                f'{" <span style=&quot;color:#15803d;font-size:11px;&quot;>(in use)</span>" if highlight else ""}</td>'
                f'<td style="padding:8px 14px;text-align:right;">{payload["top_1_accuracy"]:.1%}</td>'
                f'<td style="padding:8px 14px;text-align:right;">{payload["top_5_accuracy"]:.1%}</td>'
                f'<td style="padding:8px 14px;text-align:right;">{payload["mAP"]:.1%}</td>'
                f"</tr>"
            )
        table_html = (
            '<table style="border-collapse:collapse;margin:8px 0 16px 0;width:100%;'
            'font-size:14px;border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;">'
            '<thead style="background:#f3f4f6;color:#111827;">'
            '<tr>'
            '<th style="padding:10px 14px;text-align:left;">Backbone</th>'
            '<th style="padding:10px 14px;text-align:right;">Top-1</th>'
            '<th style="padding:10px 14px;text-align:right;">Top-5</th>'
            '<th style="padding:10px 14px;text-align:right;">mAP</th>'
            '</tr>'
            '</thead>'
            '<tbody>' + "".join(rows) + "</tbody></table>"
        )
        compare_view = mo.vstack(
            [
                mo.md("### Backbone bake-off"),
                mo.md(
                    "Same dataset and split, four feature extractors. Higher is "
                    "better on every column. The backbone currently producing the "
                    "embeddings used by this demo is highlighted in green."
                ),
                mo.Html(table_html),
            ]
        )
    return (compare_view,)


@app.cell(hide_code=True)
def _(compare_view):
    compare_view
    return


@app.cell(hide_code=True)
def _(metadata, plt):
    # --- Per-floor frame count bar chart -----------------------------------
    if metadata is None:
        per_floor_fig = None
    else:
        counts = metadata["label"].value_counts().sort_index()

        def _sort_key(label):
            text = str(label)
            if text.startswith("basement"):
                # Basements before above-ground in the chart, sorted numerically.
                try:
                    return (0, int(text.replace("basement", "")))
                except ValueError:
                    return (0, 0)
            if text.startswith("floor"):
                try:
                    return (1, int(text.replace("floor", "")))
                except ValueError:
                    return (1, 0)
            return (2, 0)

        ordered = sorted(counts.index, key=_sort_key)
        counts = counts.reindex(ordered)

        fig, ax = plt.subplots(figsize=(11, 3.5))
        _floor_colors = [
            "#a78bfa" if str(l).startswith("basement") else "#3b82f6"
            for l in counts.index
        ]
        bars = ax.bar(counts.index, counts.values, color=_floor_colors, edgecolor="white")
        ax.set_ylabel("frames")
        ax.set_title("Frames per floor — purple = basement, blue = above-ground")
        ax.tick_params(axis="x", rotation=45)
        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(counts.values) * 0.01,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#374151",
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.01)
        fig.tight_layout()
        per_floor_fig = fig
    return (per_floor_fig,)


@app.cell(hide_code=True)
def _(mo, per_floor_fig):
    if per_floor_fig is None:
        chart_view = None
    else:
        chart_view = mo.vstack([mo.md("### Frames per floor"), per_floor_fig])
    return (chart_view,)


@app.cell(hide_code=True)
def _(chart_view):
    chart_view
    return


@app.cell(hide_code=True)
def _(embeddings, metadata, mo, np, plt):
    # --- 2D PCA projection of gallery embeddings ----------------------------
    if embeddings is None or metadata is None or len(embeddings) < 3:
        pca_view = None
    else:
        # Center and run a 2-component PCA via SVD. Pure numpy.
        sample_size = min(len(embeddings), 4000)  # cap so the scatter stays readable
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(embeddings), size=sample_size, replace=False) \
            if len(embeddings) > sample_size else np.arange(len(embeddings))
        x_sample = embeddings[sample_idx]
        labels_sample = metadata["label"].astype(str).values[sample_idx]

        x_centered = x_sample - x_sample.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(x_centered, full_matrices=False)
        coords = x_centered @ vt[:2].T

        unique_labels = sorted(set(labels_sample))
        cmap = plt.get_cmap("tab20")
        color_lookup = {lbl: cmap(i % 20) for i, lbl in enumerate(unique_labels)}
        _pca_point_colors = [color_lookup[lbl] for lbl in labels_sample]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.scatter(coords[:, 0], coords[:, 1], c=_pca_point_colors, s=10, alpha=0.7, edgecolors="none")
        ax2.set_xlabel("PC 1")
        ax2.set_ylabel("PC 2")
        ax2.set_title(f"PCA of gallery embeddings ({sample_size} points, ResNet50 + L2)")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # Compact legend with one entry per floor, two columns.
        for lbl in unique_labels:
            ax2.scatter([], [], c=[color_lookup[lbl]], label=lbl, s=18)
        ax2.legend(
            ncol=2,
            fontsize=8,
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )
        fig2.tight_layout()
        pca_view = mo.vstack(
            [
                mo.md("### Embedding space (2-D PCA)"),
                mo.md(
                    "Each point is one gallery frame. Floors that cluster tightly "
                    "are easy for the model to recognise; clusters that overlap are "
                    "the hard cases the demo will struggle with."
                ),
                fig2,
            ]
        )
    return (pca_view,)


@app.cell(hide_code=True)
def _(pca_view):
    pca_view
    return


@app.cell(hide_code=True)
def _(eval_results, mo, plt):
    # --- Per-class accuracy bar chart -------------------------------------
    per_class = (eval_results or {}).get("per_class") or {}
    if not per_class:
        per_class_view = None
    else:
        def _sort_key(label):
            text = str(label)
            if text.startswith("basement"):
                try:
                    return (0, int(text.replace("basement", "")))
                except ValueError:
                    return (0, 0)
            if text.startswith("floor"):
                try:
                    return (1, int(text.replace("floor", "")))
                except ValueError:
                    return (1, 0)
            return (2, 0)

        _labels = sorted(per_class.keys(), key=_sort_key)
        _pc_top1 = [per_class[lbl]["top1_accuracy"] for lbl in _labels]
        _pc_top5 = [per_class[lbl]["top5_accuracy"] for lbl in _labels]
        _pc_mean = sum(_pc_top1) / max(len(_pc_top1), 1)

        fig_pc, ax_pc = plt.subplots(figsize=(11, 4))
        _xs = list(range(len(_labels)))
        ax_pc.bar(
            [xi - 0.2 for xi in _xs], _pc_top1, width=0.4, color="#1d4ed8", label="Top-1"
        )
        ax_pc.bar(
            [xi + 0.2 for xi in _xs], _pc_top5, width=0.4, color="#a5b4fc", label="Top-5"
        )
        ax_pc.axhline(
            _pc_mean, color="#dc2626", linestyle="--", linewidth=1,
            label=f"global Top-1 mean = {_pc_mean:.0%}",
        )
        ax_pc.set_xticks(_xs)
        ax_pc.set_xticklabels(_labels, rotation=45, ha="right", fontsize=9)
        ax_pc.set_ylim(0, 1.05)
        ax_pc.set_ylabel("accuracy")
        ax_pc.set_title("Per-class retrieval accuracy on held-out queries")
        ax_pc.spines["top"].set_visible(False)
        ax_pc.spines["right"].set_visible(False)
        ax_pc.legend(loc="upper right", fontsize=9, frameon=False)
        fig_pc.tight_layout()
        per_class_view = mo.vstack(
            [
                mo.md("### Per-class accuracy"),
                mo.md(
                    "How well the model recognises each label individually. "
                    "Bars below the dashed mean line are the failure cases the "
                    "next iteration should focus on."
                ),
                fig_pc,
            ]
        )
    return (per_class_view,)


@app.cell(hide_code=True)
def _(per_class_view):
    per_class_view
    return


@app.cell(hide_code=True)
def _(eval_results, escape, mo):
    # --- Confusion: which floors get mistaken for which -------------------
    confusion = (eval_results or {}).get("confusion_top_misses") or {}
    if not confusion:
        confusion_view = None
    else:
        # Show the 8 worst-confused source floors with their top-3
        # mistaken predictions.
        def _miss_count(rec):
            return sum(rec.values())

        ranked = sorted(confusion.items(), key=lambda kv: -_miss_count(kv[1]))[:8]
        rows_html = []
        for true_label, miss_dict in ranked:
            top_pred = sorted(miss_dict.items(), key=lambda kv: -kv[1])[:3]
            preds = ", ".join(
                f"{escape(pred)} ({n})" for pred, n in top_pred
            )
            rows_html.append(
                f'<tr>'
                f'<td style="padding:6px 12px;font-weight:600;color:#991b1b;">{escape(true_label)}</td>'
                f'<td style="padding:6px 12px;color:#374151;">{preds}</td>'
                f'</tr>'
            )
        confusion_html = (
            '<table style="border-collapse:collapse;margin:6px 0;width:100%;font-size:13px;'
            'border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;">'
            '<thead style="background:#fef2f2;color:#7f1d1d;">'
            '<tr><th style="padding:8px 12px;text-align:left;">True floor</th>'
            '<th style="padding:8px 12px;text-align:left;">Most common wrong predictions (count)</th></tr>'
            '</thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table>'
        )
        confusion_view = mo.vstack(
            [
                mo.md("### Where the model fails most"),
                mo.md(
                    "Worst-confused floors and where their queries leak to. "
                    "Look for symmetric pairs (A→B and B→A both common) — "
                    "those are floors that look architecturally identical to "
                    "ResNet50/DINOv2 because they share the same hallway / "
                    "elevator / stairs layout."
                ),
                mo.Html(confusion_html),
            ]
        )
    return (confusion_view,)


@app.cell(hide_code=True)
def _(confusion_view):
    confusion_view
    return


@app.cell(hide_code=True)
def _(mo):
    # --- "Why these numbers, and what's next?" markdown -------------------
    analysis_view = mo.md(
        """
### Analysis: why isn't accuracy higher, and what comes next?

**Where we are.** With DINOv2 ViT-S/14 at 518x518, this dataset reaches
**~52.8% Top-1 / ~72% Top-5 / ~57.7% mAP** across 25 labels (21 above-ground
floors + 4 basements). DINOv2 already beats the ResNet50 baseline by ~6 pp
on Top-1 and ~5 pp on mAP — that is the cheap win from a stronger frozen
backbone.

**Why the ceiling is here.**

1. **Vertical layout repetition.** The IE Tower repeats the same hallway,
   elevator and stairwell layout on almost every above-ground floor.
   ResNet50 and DINOv2 are not trained to read floor-number signage, so
   `floor10_hallway_left` and `floor15_hallway_left` look interchangeable
   in feature space. The per-class chart confirms this: above-ground
   floors hover around 26–55% Top-1 while the four basements (which are
   architecturally distinct) reach 77–87% Top-1.
2. **Query and gallery come from the same video.** At 1 FPS, frames from
   one continuous walkthrough are visually near-duplicates. That inflates
   Top-K accuracy — the "real" retrieval task (an unseen photo from a new
   visit) is harder than what these metrics measure. The README's failure
   analysis spells this out and recommends a second capture pass per
   floor as the most valuable data improvement.
3. **No supervised signal.** All backbones here are frozen on
   ImageNet/LVD-142M. None of them has ever seen the IE Tower, so the
   feature space is generic, not place-specific.

**Concrete improvements queued for the next iteration.** Listed in
expected-impact order:

1. **Triplet-loss fine-tuning.** Train a small projection head (or the
   last few layers of DINOv2) with positive pairs = same floor and
   negative pairs = different floor. Reuses the existing gallery as
   training data. Typical gain on similar VPR datasets: +5–15 pp Top-1.
2. **Capture a second pass per floor.** Right now every query has a
   near-duplicate in the gallery from the same video. A second, separate
   recording (different time, different phone) is the only way to make
   the metrics honest. Adds work for the team, not for the model.
3. **OCR on signage.** Adding a tiny EasyOCR / Tesseract head that
   detects floor-number plaques and routes the prediction whenever a
   number is visible would correct the bulk of vertical-symmetry errors.
4. **Test-time augmentation.** Multi-crop / multi-scale at inference,
   then average. Cheap to add, often +1–2 pp.
5. **Re-ranking with query expansion.** Initial Top-30 from FAISS, then
   re-search using the average of the top-K matches as a refined query.
   Standard trick from the image-retrieval literature.
6. **Bigger backbone / NetVLAD pooling.** Diminishing returns relative
   to (1)–(3) but useful once they are exhausted.

The plan is to merge the current state to `main` first — it's the
strongest reproducible setup we have right now — and tackle (1) and (2)
on a follow-up branch.
"""
    )
    return (analysis_view,)


@app.cell(hide_code=True)
def _(analysis_view):
    analysis_view
    return


@app.cell(hide_code=True)
def _(metadata, mo):
    # --- Section 2: Query controls -----------------------------------------
    if metadata is None:
        controls_view = None
        upload = top_k = metric = random_button = None
    else:
        upload = mo.ui.file(label="Upload a query image (JPG/PNG)", multiple=False)
        random_button = mo.ui.run_button(
            label="🎲 Pick a random held-out query", kind="success"
        )
        top_k = mo.ui.slider(1, 10, value=5, label="Top-K results", show_value=True)
        metric = mo.ui.dropdown(
            options=["cosine", "l2"], value="cosine", label="Search metric"
        )
        controls_view = mo.vstack(
            [
                mo.md("## 2. Test the model"),
                mo.md(
                    "Two options: drag a photo into the upload widget, OR press the "
                    "random button to test against a frame the model has never seen "
                    "during indexing."
                ),
                mo.hstack([upload, random_button], gap=2),
                mo.hstack([top_k, metric], gap=2),
            ]
        )
    return controls_view, metric, random_button, top_k, upload


@app.cell(hide_code=True)
def _(controls_view):
    controls_view
    return


@app.cell
def _(
    Image,
    ImageOps,
    Path,
    PROJECT_ROOT,
    bytes_to_data_uri,
    metadata,
    random,
    random_button,
    read_uploaded_image,
    upload,
):
    # Resolve the active query: uploaded image takes precedence; otherwise the
    # random button picks a held-out query frame from the metadata.
    query_image = None
    query_bytes = None
    query_filename = "query.jpg"
    query_ground_truth = None
    query_status = None

    if metadata is None:
        query_status = "_Setup required — run the pipeline first._"
    else:
        uploaded_image, uploaded_bytes, upload_message = read_uploaded_image(
            upload.value if upload is not None else None
        )
        if uploaded_image is not None and uploaded_bytes is not None:
            query_image = uploaded_image
            query_bytes = uploaded_bytes
            query_filename = upload_message
        elif random_button is not None and random_button.value:
            # The button has been clicked at least once — pick a random query.
            query_pool = (
                metadata[metadata.get("split", "") == "query"]
                if "split" in metadata.columns
                else metadata
            )
            if query_pool.empty:
                query_pool = metadata
            seed = int(random_button.value)
            row = query_pool.sample(1, random_state=seed).iloc[0]
            query_ground_truth = str(row["label"])
            image_path = Path(row["image_path"])
            for candidate in (
                image_path,
                PROJECT_ROOT / image_path,
                PROJECT_ROOT / "data" / image_path,
            ):
                if candidate.exists():
                    image_path = candidate
                    break
            try:
                query_image = ImageOps.exif_transpose(
                    Image.open(image_path)
                ).convert("RGB")
                query_bytes = image_path.read_bytes()
                query_filename = image_path.name
            except Exception:
                query_status = (
                    f"_Could not open random sample at `{image_path}`._"
                )
        else:
            query_status = upload_message

    # Stable data URI for the query preview.
    if query_image is not None and query_bytes is not None:
        query_data_uri = bytes_to_data_uri(query_bytes, query_filename)
    else:
        query_data_uri = None

    return (
        query_data_uri,
        query_filename,
        query_ground_truth,
        query_image,
        query_status,
    )


@app.cell
def _(
    device,
    extract_pil_image_embedding,
    index,
    metadata,
    metric,
    model,
    query_image,
    search_index,
    top_k,
    transform,
):
    # Run the actual retrieval if we have an active query.
    results = None
    if query_image is not None and metadata is not None and index is not None:
        embedding = extract_pil_image_embedding(
            image=query_image,
            model=model,
            transform=transform,
            device=device,
            normalize=metric.value == "cosine" if metric is not None else True,
        )
        results = search_index(
            query_embeddings=embedding,
            index=index,
            metadata=metadata,
            top_k=top_k.value if top_k is not None else 5,
            metric=metric.value if metric is not None else "cosine",
        )[0]
    return (results,)


@app.cell(hide_code=True)
def _(
    mo,
    path_to_data_uri,
    query_data_uri,
    query_ground_truth,
    query_image,
    query_status,
    render_prediction_banner,
    render_query_preview,
    render_result_card,
    results,
    summarize_prediction,
):
    if query_image is None:
        search_view = mo.md(query_status or "## Waiting for input")
    elif results is None or len(results) == 0:
        search_view = mo.md("## No results — the index returned nothing for this query.")
    else:
        predicted, votes, total = summarize_prediction(results)
        cards = [
            render_result_card(
                rank=r.rank,
                score=r.score,
                label=r.label,
                image_path=r.image_path,
                data_uri=path_to_data_uri(r.image_path),
                ground_truth=query_ground_truth,
            )
            for r in results
        ]
        gallery_html = "".join(cards)
        search_view = mo.vstack(
            [
                mo.md("### Query preview"),
                render_query_preview(mo, query_image, query_data_uri, query_ground_truth),
                mo.md("### Prediction"),
                mo.Html(render_prediction_banner(predicted, votes, total, query_ground_truth)),
                mo.md("### Top-K results"),
                mo.Html(
                    f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;">'
                    f"{gallery_html}</div>"
                ),
            ]
        )
    return (search_view,)


@app.cell(hide_code=True)
def _(search_view):
    search_view
    return


if __name__ == "__main__":
    app.run()
