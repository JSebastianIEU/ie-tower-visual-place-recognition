import marimo

__generated_with = "0.0.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import base64
    import io
    import mimetypes
    import sys
    from html import escape
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from PIL import Image

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.features.extract_embeddings import extract_pil_image_embedding
    from src.features.models import get_feature_extractor, resolve_device
    from src.features.transforms import get_image_transform
    from src.retrieval.faiss_utils import load_index
    from src.retrieval.search import search_index
    from src.utils.config import EMBEDDINGS_DIR, INDEX_DIR, ROOT

    return (
        EMBEDDINGS_DIR,
        INDEX_DIR,
        Image,
        Path,
        ROOT,
        base64,
        escape,
        extract_pil_image_embedding,
        get_feature_extractor,
        get_image_transform,
        io,
        load_index,
        mimetypes,
        mo,
        pd,
        resolve_device,
        search_index,
    )


@app.cell
def _(Image, Path, base64, escape, io, mimetypes):
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
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception:
            return None, None, "The uploaded file is not a valid image."

        return image, payload, filename or "query.jpg"

    def bytes_to_data_uri(data: bytes, filename: str) -> str:
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "image/jpeg"
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def path_to_data_uri(image_path: str) -> str | None:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return None
        return bytes_to_data_uri(path.read_bytes(), path.name)

    def render_result_card(rank, score, label, image_path, data_uri):
        label_text = escape(label) if label else "N/A"
        path_text = escape(image_path) if image_path else "N/A"
        image_html = (
            f'<img src="{data_uri}" alt="Retrieved result {rank}" '
            'style="width:100%;height:180px;object-fit:cover;border-radius:8px;" />'
            if data_uri
            else (
                '<div style="height:180px;display:flex;align-items:center;'
                'justify-content:center;background:#f3f4f6;border-radius:8px;'
                'color:#6b7280;">Preview unavailable</div>'
            )
        )

        return f"""
        <div style="border:1px solid #d1d5db;border-radius:12px;padding:12px;background:#ffffff;">
          <div style="font-size:14px;font-weight:600;margin-bottom:8px;">Rank {rank}</div>
          {image_html}
          <div style="margin-top:10px;font-size:14px;"><strong>Score:</strong> {score:.4f}</div>
          <div style="font-size:14px;"><strong>Label:</strong> {label_text}</div>
          <div style="font-size:12px;color:#4b5563;word-break:break-word;margin-top:6px;">{path_text}</div>
        </div>
        """

    def render_query_preview(mo, image, data_uri):
        image_widget = getattr(mo, "image", None)
        if image_widget is not None:
            try:
                return image_widget(image)
            except Exception:
                pass

        return mo.Html(
            f'<img src="{data_uri}" alt="Query image" '
            'style="max-width:320px;width:100%;border-radius:12px;border:1px solid #d1d5db;" />'
        )

    return (
        bytes_to_data_uri,
        path_to_data_uri,
        read_uploaded_image,
        render_query_preview,
        render_result_card,
    )


@app.cell(hide_code=True)
def _(mo):
    upload = mo.ui.file(label="Upload a query image", multiple=False)
    top_k = mo.ui.slider(1, 10, value=5, label="Top-K results")
    metric = mo.ui.dropdown(
        options=["cosine", "l2"],
        value="cosine",
        label="Search metric",
    )
    return metric, top_k, upload


@app.cell(hide_code=True)
def _(EMBEDDINGS_DIR, INDEX_DIR, ROOT, metric, mo, top_k, upload):
    expected_index = INDEX_DIR / "gallery.index"
    expected_metadata = EMBEDDINGS_DIR / "gallery_metadata.csv"

    instructions = mo.md(
        f"""
        # IE Tower VPR Demo

        This app runs a query against the existing FAISS retrieval pipeline.
        It expects:

        - `{expected_index}`
        - `{expected_metadata}`
        - project root: `{ROOT}`

        Build them first with `python scripts/run_pipeline.py`
        or with `python scripts/extract_embeddings.py` followed by `python scripts/build_index.py`.
        """
    )

    controls_view = mo.vstack([instructions, upload, top_k, metric])
    return controls_view, expected_index, expected_metadata


@app.cell(hide_code=True)
def _(controls_view):
    controls_view
    return


@app.cell
def _(expected_index, expected_metadata, load_index, pd):
    errors = []
    index = None
    metadata = None

    if not expected_index.exists():
        errors.append(
            f"Missing FAISS index: {expected_index}. Run python scripts/build_index.py first."
        )
    else:
        index = load_index(expected_index)

    if not expected_metadata.exists():
        errors.append(
            "Missing metadata CSV: "
            f"{expected_metadata}. Run python scripts/extract_embeddings.py first."
        )
    else:
        metadata = pd.read_csv(expected_metadata)

    return errors, index, metadata


@app.cell
def _(get_feature_extractor, get_image_transform, resolve_device):
    device = resolve_device(None)
    model, _ = get_feature_extractor("resnet50")
    transform = get_image_transform("resnet50")
    return device, model, transform


@app.cell(hide_code=True)
def _(
    bytes_to_data_uri,
    device,
    errors,
    extract_pil_image_embedding,
    index,
    metadata,
    metric,
    mo,
    model,
    path_to_data_uri,
    read_uploaded_image,
    render_query_preview,
    render_result_card,
    search_index,
    top_k,
    transform,
    upload,
):
    search_view = None

    if errors:
        error_list = "\n".join(f"- {message}" for message in errors)
        search_view = mo.md(f"## Setup required\n\n{error_list}")
        return (search_view,)

    uploaded_image, uploaded_bytes, upload_name_or_message = read_uploaded_image(upload.value)
    if uploaded_image is None or uploaded_bytes is None:
        search_view = mo.md(f"## Waiting for input\n\n{upload_name_or_message}")
        return (search_view,)

    query_embedding = extract_pil_image_embedding(
        image=uploaded_image,
        model=model,
        transform=transform,
        device=device,
        normalize=metric.value == "cosine",
    )
    results = search_index(
        query_embeddings=query_embedding,
        index=index,
        metadata=metadata,
        top_k=top_k.value,
        metric=metric.value,
    )[0]

    query_preview = bytes_to_data_uri(uploaded_bytes, upload_name_or_message)
    cards = []
    for result in results:
        cards.append(
            render_result_card(
                rank=result.rank,
                score=result.score,
                label=result.label,
                image_path=result.image_path,
                data_uri=path_to_data_uri(result.image_path),
            )
        )

    if not cards:
        search_view = mo.md("## No results\n\nThe index returned no matches for this query.")
        return (search_view,)

    gallery_html = "".join(cards)
    search_view = mo.vstack(
        [
            mo.md("## Query preview"),
            render_query_preview(mo, uploaded_image, query_preview),
            mo.md("## Retrieved results"),
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
