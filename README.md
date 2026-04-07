# IE Tower Visual Place Recognition

## Project Overview

This repository contains a modular Visual Place Recognition (VPR) pipeline for the IE Tower project. It loads labeled place images, extracts image embeddings with a pretrained CNN, stores the embeddings on disk, builds a FAISS index for fast retrieval, and evaluates retrieval quality with Top-K accuracy and mAP.

The interactive layer uses `marimo`, while the core pipeline remains regular Python modules and scripts under `src/` and `scripts/`. The codebase is intentionally small and modular so the data, feature extraction, retrieval, and evaluation components can evolve independently.

## Repository Structure

```text
ie-tower-visual-place-recognition/
├── app/
│   └── demo.py
├── data/
│   ├── metadata/
│   │   └── dataset.csv
│   ├── processed_frames/
│   └── raw_videos/
├── outputs/
│   ├── embeddings/
│   ├── index/
│   └── results/
├── scripts/
│   ├── build_index.py
│   ├── extract_embeddings.py
│   ├── run_evaluation.py
│   ├── run_pipeline.py
│   └── run_query.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── retrieval/
│   └── utils/
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset is not included in this repository. You must provide your own images and metadata.

Expected metadata file:

- `data/metadata/dataset.csv`

Required columns:

- `image_path`: relative or absolute path to the image
- `label`: place identifier used as the ground-truth location

Recommended optional columns:

- `split`: use values like `gallery` and `query` for evaluation
- `device`: capture device metadata
- `lighting`: lighting condition metadata

Example:

```csv
image_path,label,split,device,lighting
processed_frames/floor1/img_0001.jpg,floor1,gallery,iphone,bright
processed_frames/floor1/img_0002.jpg,floor1,query,iphone,dim
processed_frames/floor2/img_0003.jpg,floor2,gallery,gopro,bright
```

Keep labels consistent across gallery and query images. For example, if one location is labeled `floor1`, do not mix it with variants like `Floor 1` or `first_floor`.

## Default Paths

- Frames: `data/processed_frames/`
- Embeddings: `outputs/embeddings/`
- FAISS index: `outputs/index/`

## Pipeline Steps

1. Prepare the dataset metadata in `data/metadata/dataset.csv` and place images under `data/processed_frames/` or another path referenced by the CSV.
2. Extract embeddings:

```bash
python scripts/extract_embeddings.py
```

3. Build the FAISS index:

```bash
python scripts/build_index.py
```

4. Run a single query:

```bash
python scripts/run_query.py --image path/to/image.jpg
```

5. Evaluate retrieval quality:

```bash
python scripts/run_evaluation.py
```

6. Run the default end-to-end pipeline:

```bash
python scripts/run_pipeline.py
```

Notes on the scripts:

- The scripts use sensible default paths under `data/` and `outputs/`.
- You can override defaults with CLI flags such as `--csv-path`, `--image-root`, `--index-path`, and `--metadata-path`.
- `run_query.py` expects that embeddings and the FAISS index have already been generated.

## Running the Marimo Demo

After building embeddings and the FAISS index, start the demo app with:

```bash
marimo run app/demo.py
```

The demo:

- lets you upload a query image
- uses the existing `src/features` and `src/retrieval` modules
- searches the built FAISS index
- displays the top-K retrieved results and labels when available
- shows clear setup messages if the index or metadata files are missing

By default, the demo looks for:

- `outputs/index/gallery.index`
- `outputs/embeddings/gallery_metadata.csv`

## Outputs

Generated files are saved under `outputs/`:

- `outputs/embeddings/`
  Saved `.npy` embedding arrays and metadata CSV files.
- `outputs/index/`
  Saved FAISS index files such as `gallery.index`.
- `outputs/results/`
  Saved evaluation results such as `evaluation.json`.

## Notes

- The dataset is not included.
- The code is modular: `src/data`, `src/features`, `src/retrieval`, and `src/evaluation` are kept separate on purpose.
- The current baseline uses a pretrained `ResNet50` and FAISS.
- The interactive/demo layer uses `marimo` only. The main pipeline remains regular Python scripts.
- If the dataset or outputs are missing, the scripts and demo will not invent results.

## License

This project is distributed under the license in [LICENSE](LICENSE).
